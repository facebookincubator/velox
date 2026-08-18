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
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include <fmt/format.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

class DeltaBlockEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> toVector(const std::vector<T>& input) {
    nimble::Vector<T> values{pool_.get()};
    values.insert(values.end(), input.data(), input.data() + input.size());
    return values;
  }

  template <typename T>
  std::unique_ptr<nimble::Encoding> createEncoding(
      const std::vector<T>& input,
      const nimble::Encoding::Options& options = {.deltaBlockSize = 3}) {
    auto values = toVector(input);
    return nimble::test::Encoder<nimble::DeltaBlockEncoding<T>>::createEncoding(
        *buffer_,
        values,
        nullptr,
        nimble::CompressionType::Uncompressed,
        options);
  }

  template <typename T>
  void roundTrip(const std::vector<T>& input) {
    auto encoding = createEncoding(input);
    std::vector<T> output(input.size());
    encoding->materialize(static_cast<uint32_t>(input.size()), output.data());

    EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::DeltaBlock);
    EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<T>::dataType);
    EXPECT_EQ(encoding->rowCount(), static_cast<uint32_t>(input.size()));
    EXPECT_EQ(output, input);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
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

  explicit IntegralReadWithVisitor(std::vector<velox::vector_size_t> rows)
      : rows_{std::move(rows)} {}

  TestReader& reader() {
    return reader_;
  }

  velox::vector_size_t numRows() const {
    return rows_.size();
  }

  velox::vector_size_t rowAt(velox::vector_size_t index) const {
    return rows_[index];
  }

  velox::vector_size_t currentRow() const {
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

  void addRowIndex(velox::vector_size_t count) {
    rowIndex_ += count;
  }

  void addNumValues(velox::vector_size_t /*count*/) {}

  bool atEnd() const {
    return rowIndex_ >= rows_.size();
  }

  const std::vector<T>& values() const {
    return values_;
  }

 private:
  TestReader reader_;
  std::vector<velox::vector_size_t> rows_;
  velox::vector_size_t rowIndex_{0};
  std::vector<T> values_;
};

TEST_F(DeltaBlockEncodingTest, roundTripUnsignedValues) {
  roundTrip<uint32_t>({1, 2, 3, 7, 9, 12, 13, 15, 20, 100});
  roundTrip<uint64_t>({0, 0, 0, 5, 5, 9, 1000, 2000, 2000, 4000});
  roundTrip<uint16_t>({2, 4, 4, 4, 9, 12, 30});
  roundTrip<uint8_t>({0, 1, 1, 2, 3, 8});
}

TEST_F(DeltaBlockEncodingTest, roundTripSignedValues) {
  roundTrip<int32_t>({-5, -3, -3, -1, 0, 2, 9, 10});
  roundTrip<int64_t>({-100, -100, -7, -1, 0, 1, 1000000});
  roundTrip<int16_t>({-10, -5, -5, 0, 2, 4});
  roundTrip<int8_t>({-8, -2, -2, 0, 5, 9});
}

TEST_F(DeltaBlockEncodingTest, resetSkipAndMaterialize) {
  const std::vector<uint32_t> input{0, 1, 2, 10, 11, 12, 100, 101, 102};
  auto encoding = createEncoding(input);

  encoding->skip(4);
  std::vector<uint32_t> output(3);
  encoding->materialize(static_cast<uint32_t>(output.size()), output.data());
  EXPECT_EQ(output, (std::vector<uint32_t>{11, 12, 100}));

  encoding->reset();
  output.resize(input.size());
  encoding->materialize(static_cast<uint32_t>(output.size()), output.data());
  EXPECT_EQ(output, input);
}

TEST_F(DeltaBlockEncodingTest, readWithVisitorDenseAndSparseAcrossBlocks) {
  std::vector<int64_t> input;
  input.reserve(257);
  int64_t value{-1000};
  for (uint32_t i = 0; i < 257; ++i) {
    value += i % 11;
    input.push_back(value);
  }

  auto encoding = createEncoding<int64_t>(input, {.deltaBlockSize = 32});

  struct TestCase {
    std::string name;
    std::vector<velox::vector_size_t> rows;
  };

  const std::vector<TestCase> testCases{
      {.name = "dense", .rows = {0, 1, 2, 3, 4, 5, 6, 7}},
      {.name = "sparse across blocks", .rows = {0, 31, 32, 33, 127, 256}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    encoding->reset();
    IntegralReadWithVisitor<int64_t> visitor{testCase.rows};
    nimble::ReadWithVisitorParams params;
    params.numScanned = 0;

    dynamic_cast<nimble::DeltaBlockEncoding<int64_t>&>(*encoding)
        .readWithVisitor(visitor, params);

    ASSERT_EQ(visitor.values().size(), testCase.rows.size());
    for (size_t i = 0; i < testCase.rows.size(); ++i) {
      EXPECT_EQ(visitor.values()[i], input[testCase.rows[i]]);
    }
  }
}

TEST_F(DeltaBlockEncodingTest, randomizedRoundTrip) {
  struct TestCase {
    uint32_t seed;
    uint32_t rowCount;
    uint16_t blockSize;
    int64_t start;
    int64_t maxStep;
  };

  const std::vector<TestCase> testCases{
      {.seed = 1, .rowCount = 1, .blockSize = 1, .start = -100, .maxStep = 0},
      {.seed = 2, .rowCount = 37, .blockSize = 3, .start = -1000, .maxStep = 5},
      {.seed = 3, .rowCount = 257, .blockSize = 64, .start = 0, .maxStep = 19},
      {.seed = 4,
       .rowCount = 1025,
       .blockSize = 256,
       .start = 1'000'000,
       .maxStep = 1024},
      {.seed = 5,
       .rowCount = 4097,
       .blockSize = 1024,
       .start = std::numeric_limits<int32_t>::max(),
       .maxStep = 4096},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "seed={}, rowCount={}, blockSize={}",
            testCase.seed,
            testCase.rowCount,
            testCase.blockSize));
    std::mt19937 rng{testCase.seed};
    std::uniform_int_distribution<int64_t> stepDist{0, testCase.maxStep};

    std::vector<int64_t> input;
    input.reserve(testCase.rowCount);
    int64_t value{testCase.start};
    for (uint32_t i = 0; i < testCase.rowCount; ++i) {
      value += stepDist(rng);
      input.push_back(value);
    }

    auto encoding =
        createEncoding<int64_t>(input, {.deltaBlockSize = testCase.blockSize});
    std::vector<int64_t> output(input.size());
    uint32_t offset{0};
    while (offset < input.size()) {
      const auto batchSize =
          std::min<uint32_t>(1 + (rng() % 97), input.size() - offset);
      encoding->materialize(batchSize, output.data() + offset);
      offset += batchSize;
    }
    EXPECT_EQ(output, input);
  }
}

TEST_F(DeltaBlockEncodingTest, estimateRejectsUnsortedValues) {
  const std::vector<uint32_t> input{1, 5, 4, 8};
  const auto values = toVector(input);
  const auto physicalValues = std::span<const uint32_t>{
      reinterpret_cast<const uint32_t*>(values.data()), values.size()};

  EXPECT_EQ(
      nimble::DeltaBlockEncoding<uint32_t>::estimateSize(
          physicalValues, {.deltaBlockSize = 3}),
      std::nullopt);
}

TEST_F(DeltaBlockEncodingTest, estimateUsesVarintPrefixSize) {
  const std::vector<uint32_t> input{1, 2, 5, 9, 10};
  const auto values = toVector(input);
  const auto physicalValues = std::span<const uint32_t>{
      reinterpret_cast<const uint32_t*>(values.data()), values.size()};

  const auto fixedPrefixSize =
      nimble::DeltaBlockEncoding<uint32_t>::estimateSize(
          physicalValues, {.useVarintRowCount = false, .deltaBlockSize = 3});
  const auto varintPrefixSize =
      nimble::DeltaBlockEncoding<uint32_t>::estimateSize(
          physicalValues, {.useVarintRowCount = true, .deltaBlockSize = 3});

  ASSERT_TRUE(fixedPrefixSize.has_value());
  ASSERT_TRUE(varintPrefixSize.has_value());
  EXPECT_EQ(fixedPrefixSize.value(), varintPrefixSize.value());
}

TEST_F(DeltaBlockEncodingTest, encodeRejectsUnsortedValues) {
  const std::vector<int32_t> input{-1, -2};
  EXPECT_THROW(createEncoding(input), nimble::NimbleException);
}
