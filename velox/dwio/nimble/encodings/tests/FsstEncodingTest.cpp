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
#include "velox/dwio/nimble/encodings/FsstEncoding.h"

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble::test {
namespace {

class FsstEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    if (!velox::memory::MemoryManager::testInstance()) {
      velox::memory::MemoryManager::initialize({});
    }
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
  }

  std::unique_ptr<EncodingSelectionPolicy<std::string_view>>
  createSelectionPolicy(
      CompressionOptions compressionOptions = {},
      CompressionType compressionType = CompressionType::Uncompressed) {
    // FSST has one nested child encoding for compressed string lengths.
    std::vector<std::optional<const EncodingLayout>> children;
    children.emplace_back(
        EncodingLayout{
            EncodingType::Trivial, {}, CompressionType::Uncompressed});
    EncodingLayout layout{
        EncodingType::Fsst, {}, compressionType, std::move(children)};
    return std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
        std::move(layout),
        std::move(compressionOptions),
        encodingSelectionPolicyCreator_);
  }

  std::unique_ptr<EncodingSelectionPolicy<std::string_view>>
  createTrivialSelectionPolicy(CompressionOptions compressionOptions) {
    const auto compressionType = compressionOptions.compressionType;
    std::vector<std::optional<const EncodingLayout>> children;
    children.emplace_back(
        EncodingLayout{
            EncodingType::Trivial, {}, CompressionType::Uncompressed});
    EncodingLayout layout{
        EncodingType::Trivial, {}, compressionType, std::move(children)};
    return std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
        std::move(layout),
        std::move(compressionOptions),
        encodingSelectionPolicyCreator_);
  }

  std::function<void*(uint32_t)> createStringBufferFactory() {
    return [this](uint32_t totalLength) {
      auto& buffer = stringBuffers_.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
      return buffer->asMutable<void>();
    };
  }

  std::string_view encodeFsst(
      const std::vector<std::string_view>& values,
      Buffer& buffer) {
    return EncodingFactory::encode<std::string_view>(
        createSelectionPolicy(),
        values,
        buffer,
        {.fsstCompressionTargetRatio = std::numeric_limits<double>::max()});
  }

  void roundTrip(
      const std::vector<std::string_view>& values,
      const std::string& testName = "",
      EncodingType expectedEncodingType = EncodingType::Fsst) {
    SCOPED_TRACE(testName);

    Buffer buffer{*pool_};
    auto encoded = encodeFsst(values, buffer);
    stringBuffers_.clear();

    auto encoding = EncodingFactory().create(
        *pool_, encoded, createStringBufferFactory());

    ASSERT_EQ(encoding->dataType(), DataType::String);
    ASSERT_EQ(encoding->encodingType(), expectedEncodingType);
    ASSERT_EQ(encoding->rowCount(), values.size());

    std::vector<std::string_view> decoded(values.size());
    encoding->materialize(values.size(), decoded.data());

    ASSERT_EQ(decoded.size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(decoded[i], values[i]) << "mismatch at row " << i;
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::vector<velox::BufferPtr> stringBuffers_;
  ManualEncodingSelectionPolicyFactory manualPolicyFactory_;
  EncodingSelectionPolicyCreator encodingSelectionPolicyCreator_ =
      [this](DataType dataType) {
        return manualPolicyFactory_.createPolicy(dataType);
      };
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

class StringReadWithVisitor {
 public:
  using DataType = std::string_view;
  using Extract = std::nullptr_t;

  explicit StringReadWithVisitor(std::vector<vector_size_t> rows)
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

  void process(std::string_view value, bool& atEnd) {
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

  void addNumValues(vector_size_t /* count */) {}

  bool atEnd() const {
    return rowIndex_ >= rows_.size();
  }

  const std::vector<std::string_view>& values() const {
    return values_;
  }

 private:
  TestReader reader_;
  std::vector<vector_size_t> rows_;
  vector_size_t rowIndex_{0};
  std::vector<std::string_view> values_;
};

TEST_F(FsstEncodingTest, roundTripStrings) {
  struct TestCase {
    std::string name;
    std::vector<std::string_view> values;
    EncodingType expectedEncodingType{EncodingType::Fsst};
  };

  std::vector<std::string> highCardinalityStorage;
  highCardinalityStorage.reserve(1'000);
  for (int i = 0; i < 1'000; ++i) {
    highCardinalityStorage.emplace_back(
        fmt::format("url/path/segment/{}/page?id={}", i, i));
  }
  std::vector<std::string_view> highCardinalityValues;
  highCardinalityValues.reserve(highCardinalityStorage.size());
  for (const auto& value : highCardinalityStorage) {
    highCardinalityValues.emplace_back(value);
  }

  std::string stringWithNullBytes1("ab\x00\x01\x02", 5);
  std::string stringWithNullBytes2("\x00\xff\x00", 3);
  std::string normalString("normal");
  std::string longString1(10'000, 'a');
  std::string longString2(10'000, 'b');

  std::vector<TestCase> testCases{
      {"basic strings", {"hello", "world", "hello world", "foo", "bar", "baz"}},
      {"all empty strings", {"", "", "", ""}, EncodingType::Trivial},
      {"mixed empty and non-empty", {"", "abc", "", "def", ""}},
      {"single string", {"single"}},
      {"high cardinality URLs", std::move(highCardinalityValues)},
      {"strings with null bytes",
       {std::string_view{stringWithNullBytes1},
        std::string_view{stringWithNullBytes2},
        std::string_view{normalString}}},
      {"long strings",
       {std::string_view{longString1}, std::string_view{longString2}}},
  };

  for (const auto& testCase : testCases) {
    roundTrip(testCase.values, testCase.name, testCase.expectedEncodingType);
  }
}

TEST_F(FsstEncodingTest, slice) {
  const std::vector<std::string> storage{
      "common/prefix/value/0000",
      "common/prefix/value/0001",
      "common/prefix/value/0002",
      "common/prefix/value/0003",
      "common/prefix/value/0004",
      "common/prefix/value/0005",
  };
  std::vector<std::string_view> values;
  values.reserve(storage.size());
  for (const auto& value : storage) {
    values.push_back(value);
  }

  Buffer buffer{*pool_};
  const auto encoded = encodeFsst(values, buffer);
  ASSERT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Fsst);

  Buffer sliceBuffer{*pool_};
  const auto sliced = EncodingFactory::slice(
      encoded,
      /*offset=*/2,
      /*length=*/3,
      sliceBuffer,
      {.fsstCompressionTargetRatio = std::numeric_limits<double>::max()});

  auto encoding =
      EncodingFactory().create(*pool_, sliced, createStringBufferFactory());
  ASSERT_EQ(encoding->encodingType(), EncodingType::Fsst);
  ASSERT_EQ(encoding->dataType(), DataType::String);
  ASSERT_EQ(encoding->rowCount(), 3);

  std::vector<std::string_view> decoded(3);
  encoding->materialize(decoded.size(), decoded.data());

  const std::vector<std::string_view> expected{values[2], values[3], values[4]};
  EXPECT_EQ(decoded, expected);
}

TEST_F(FsstEncodingTest, capturesLengthsLayoutWithVarintHeaderSizes) {
  std::vector<std::string> storage;
  storage.reserve(512);
  for (uint32_t i = 0; i < 512; ++i) {
    storage.emplace_back(
        fmt::format("common/prefix/for/fsst/layout/{:04}", i % 97));
  }

  std::vector<std::string_view> values;
  values.reserve(storage.size());
  for (const auto& value : storage) {
    values.push_back(value);
  }

  Buffer buffer{*pool_};
  const auto encoded = encodeFsst(values, buffer);
  ASSERT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Fsst);

  const char* pos =
      encoded.data() + EncodingPrefix::prefixSize(encoded, /*useVarint=*/false);
  const auto symbolTableSize = varint::readVarint32(&pos);
  ASSERT_GT(symbolTableSize, 0);
  pos += symbolTableSize;
  const auto lengthsSize = varint::readVarint32(&pos);
  ASSERT_GT(lengthsSize, 0);

  const auto lengths = FsstEncoding::lengthsEncoding(encoded);
  EXPECT_EQ(lengths.data(), pos);
  EXPECT_EQ(lengths.size(), lengthsSize);

  const auto captured = EncodingLayoutCapture::capture(encoded, {});
  ASSERT_EQ(captured.encodingType(), EncodingType::Fsst);
  ASSERT_EQ(captured.childrenCount(), 1);
  ASSERT_TRUE(captured.child(EncodingIdentifiers::Fsst::Lengths).has_value());
  EXPECT_EQ(
      captured.child(EncodingIdentifiers::Fsst::Lengths)->encodingType(),
      EncodingType::Trivial);
}

TEST_F(FsstEncodingTest, invalidSliceRange) {
  const std::vector<std::string> storage{
      "common/prefix/value/0000",
      "common/prefix/value/0001",
      "common/prefix/value/0002",
      "common/prefix/value/0003",
  };
  std::vector<std::string_view> values;
  values.reserve(storage.size());
  for (const auto& value : storage) {
    values.push_back(value);
  }

  Buffer buffer{*pool_};
  const auto encoded = encodeFsst(values, buffer);

  Buffer sliceBuffer{*pool_};
  NIMBLE_ASSERT_THROW(
      FsstEncoding::slice(
          encoded,
          /*offset=*/0,
          /*length=*/0,
          sliceBuffer,
          Encoding::Options{}),
      "Cannot slice zero rows.");
  NIMBLE_ASSERT_THROW(
      FsstEncoding::slice(
          encoded,
          /*offset=*/values.size(),
          /*length=*/1,
          sliceBuffer,
          Encoding::Options{}),
      "");
}

TEST_F(FsstEncodingTest, sliceRandomRanges) {
  std::vector<std::string> storage;
  storage.reserve(256);
  for (uint32_t i = 0; i < 256; ++i) {
    storage.emplace_back(
        fmt::format("common/prefix/for/random/fsst/range/{:04}", i % 37));
  }
  std::vector<std::string_view> values;
  values.reserve(storage.size());
  for (const auto& value : storage) {
    values.push_back(value);
  }

  Buffer buffer{*pool_};
  const auto encoded = encodeFsst(values, buffer);
  ASSERT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Fsst);

  std::mt19937 rng{0x5eed};
  const auto rowCount = static_cast<uint32_t>(values.size());
  for (uint32_t iteration = 0; iteration < 64; ++iteration) {
    const auto offset =
        std::uniform_int_distribution<uint32_t>{0, rowCount - 1}(rng);
    const auto length =
        std::uniform_int_distribution<uint32_t>{1, rowCount - offset}(rng);
    SCOPED_TRACE(
        testing::Message() << "iteration=" << iteration << ", offset=" << offset
                           << ", length=" << length);

    Buffer sliceBuffer{*pool_};
    const auto sliced = EncodingFactory::slice(
        encoded,
        offset,
        length,
        sliceBuffer,
        {.fsstCompressionTargetRatio = std::numeric_limits<double>::max()});

    stringBuffers_.clear();
    auto encoding =
        EncodingFactory().create(*pool_, sliced, createStringBufferFactory());
    ASSERT_EQ(encoding->encodingType(), EncodingType::Fsst);
    ASSERT_EQ(encoding->rowCount(), length);

    std::vector<std::string_view> decoded(length);
    encoding->materialize(length, decoded.data());
    for (uint32_t i = 0; i < length; ++i) {
      ASSERT_EQ(decoded[i], values[offset + i]) << "row " << i;
    }
  }
}

TEST_F(FsstEncodingTest, estimateSizeUsesTargetRatioAndLengthEncodingWidth) {
  const std::vector<std::string_view> values{
      "alpha",
      "common/prefix/string/value/one",
      "common/prefix/string/value/two",
      "common/prefix/string/value/three",
  };
  const auto statistics = Statistics<std::string_view>::create(values);
  auto makeOptions = [](double compressionTargetRatio) {
    Encoding::Options options;
    options.fsstCompressionTargetRatio = compressionTargetRatio;
    return options;
  };

  auto estimatedVariablePart = [&](const Encoding::Options& options) {
    const uint64_t estimatedBlobSize = static_cast<uint64_t>(
        statistics.totalStringsLength() * options.fsstCompressionTargetRatio);
    const uint64_t estimatedMaxCompressedLength =
        static_cast<uint64_t>(std::ceil(
            statistics.max().size() * options.fsstCompressionTargetRatio));
    return estimatedBlobSize +
        FixedBitWidthEncoding<uint32_t>::estimateSize(
               values.size(), 0, estimatedMaxCompressedLength, options);
  };

  const auto optionsAt50 = makeOptions(0.5);
  const auto optionsAt60 = makeOptions(0.6);
  const auto fixedWidthSizeAt50 =
      FsstEncoding::estimateSize(values.size(), statistics, optionsAt50);
  const auto fixedWidthSizeAt60 =
      FsstEncoding::estimateSize(values.size(), statistics, optionsAt60);

  EXPECT_EQ(
      fixedWidthSizeAt60 - fixedWidthSizeAt50,
      estimatedVariablePart(optionsAt60) - estimatedVariablePart(optionsAt50));
}

TEST_F(FsstEncodingTest, fsstCompressionTargetRatioFallsBackToTrivial) {
  std::vector<std::string> storage;
  storage.reserve(1'000);
  for (int i = 0; i < 1'000; ++i) {
    storage.emplace_back(
        fmt::format(
            "common/prefix/with/repeated/symbols/{:04}/common/suffix", i));
  }

  std::vector<std::string_view> values;
  values.reserve(storage.size());
  size_t totalRawSize{0};
  for (const auto& value : storage) {
    values.emplace_back(value);
    totalRawSize += value.size();
  }

  Buffer fsstOnlyBuffer{*pool_};
  const auto fsstOnlyEncoded = EncodingFactory::encode<std::string_view>(
      createSelectionPolicy(),
      values,
      fsstOnlyBuffer,
      {.fsstCompressionTargetRatio = std::numeric_limits<double>::max()});

  Buffer trivialOnlyBuffer{*pool_};
  CompressionOptions noCompressionOptions;
  noCompressionOptions.compressionType = CompressionType::Uncompressed;
  const auto trivialOnlyEncoded = EncodingFactory::encode<std::string_view>(
      createTrivialSelectionPolicy(noCompressionOptions),
      values,
      trivialOnlyBuffer);

  const auto permissiveTargetRatio =
      static_cast<double>(fsstOnlyEncoded.size()) / totalRawSize;

  Buffer permissiveTargetBuffer{*pool_};
  auto permissiveTargetEncoded = EncodingFactory::encode<std::string_view>(
      createSelectionPolicy(),
      values,
      permissiveTargetBuffer,
      {.fsstCompressionTargetRatio = permissiveTargetRatio});
  stringBuffers_.clear();

  auto permissiveTargetEncoding = EncodingFactory().create(
      *pool_, permissiveTargetEncoded, createStringBufferFactory());
  EXPECT_EQ(permissiveTargetEncoding->encodingType(), EncodingType::Fsst);

  Buffer strictTargetBuffer{*pool_};
  auto strictTargetEncoded = EncodingFactory::encode<std::string_view>(
      createSelectionPolicy(),
      values,
      strictTargetBuffer,
      {.fsstCompressionTargetRatio = 0});
  stringBuffers_.clear();

  auto strictTargetEncoding = EncodingFactory().create(
      *pool_, strictTargetEncoded, createStringBufferFactory());

  EXPECT_EQ(strictTargetEncoded, trivialOnlyEncoded);
  EXPECT_EQ(strictTargetEncoding->encodingType(), EncodingType::Trivial);
  std::vector<std::string_view> decoded(values.size());
  strictTargetEncoding->materialize(values.size(), decoded.data());
  EXPECT_EQ(decoded, values);
}

TEST_F(FsstEncodingTest, skipAndMaterialize) {
  std::vector<std::string_view> values = {
      "alpha", "bravo", "charlie", "delta", "echo"};

  Buffer buffer{*pool_};
  auto encoded = encodeFsst(values, buffer);
  stringBuffers_.clear();

  auto encoding =
      EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

  encoding->skip(2);

  std::vector<std::string_view> decoded(2);
  encoding->materialize(2, decoded.data());

  EXPECT_EQ(decoded[0], "charlie");
  EXPECT_EQ(decoded[1], "delta");
}

TEST_F(FsstEncodingTest, readWithVisitorReadsSelectedRows) {
  std::vector<std::string_view> values = {
      "zero/value",
      "one/value",
      "two/value",
      "three/value",
      "four/value",
      "five/value",
  };

  Buffer buffer{*pool_};
  const auto encoded = encodeFsst(values, buffer);

  struct TestCase {
    std::string name;
    std::vector<vector_size_t> rows;
    std::vector<std::string_view> expected;
  };

  for (const auto& testCase : std::vector<TestCase>{
           {
               .name = "dense rows",
               .rows = {0, 1, 2, 3, 4, 5},
               .expected = values,
           },
           {
               .name = "sparse rows",
               .rows = {0, 2, 5},
               .expected = {values[0], values[2], values[5]},
           },
       }) {
    SCOPED_TRACE(testCase.name);

    stringBuffers_.clear();
    FsstEncoding encoding{*pool_, encoded, createStringBufferFactory()};
    StringReadWithVisitor visitor{testCase.rows};
    ReadWithVisitorParams params;
    params.numScanned = 0;

    encoding.readWithVisitor(visitor, params);

    EXPECT_EQ(visitor.values(), testCase.expected);
  }
}

TEST_F(FsstEncodingTest, resetAndRematerialize) {
  std::vector<std::string_view> values = {"first", "second", "third"};

  Buffer buffer{*pool_};
  auto encoded = encodeFsst(values, buffer);
  stringBuffers_.clear();

  auto encoding =
      EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

  std::vector<std::string_view> decoded1(3);
  encoding->materialize(3, decoded1.data());

  encoding->reset();

  std::vector<std::string_view> decoded2(3);
  encoding->materialize(3, decoded2.data());

  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(decoded1[i], values[i]);
    EXPECT_EQ(decoded2[i], values[i]);
  }
}

TEST_F(FsstEncodingTest, materializeOneAtATime) {
  std::vector<std::string_view> values = {"alpha", "bravo", "charlie"};

  Buffer buffer{*pool_};
  auto encoded = encodeFsst(values, buffer);
  stringBuffers_.clear();

  auto encoding =
      EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

  for (size_t i = 0; i < values.size(); ++i) {
    std::string_view decoded;
    encoding->materialize(1, &decoded);
    EXPECT_EQ(decoded, values[i]) << "mismatch at row " << i;
  }
}

TEST_F(FsstEncodingTest, compressionRatio) {
  std::vector<std::string> storage;
  std::vector<std::string_view> values;
  storage.reserve(1'000);
  for (int i = 0; i < 1'000; ++i) {
    storage.emplace_back(
        fmt::format("https://www.example.com/api/v2/users/{}/profile", i));
  }
  values.reserve(storage.size());
  for (const auto& s : storage) {
    values.emplace_back(s);
  }

  Buffer fsstBuffer{*pool_};
  auto fsstEncoded = encodeFsst(values, fsstBuffer);

  size_t totalRawSize = 0;
  for (const auto& v : values) {
    totalRawSize += v.size();
  }

  EXPECT_LT(fsstEncoded.size(), totalRawSize)
      << "FSST should compress the data";
}

TEST_F(FsstEncodingTest, debugString) {
  std::vector<std::string> storage;
  std::vector<std::string_view> values;
  storage.reserve(2'000);
  for (int i = 0; i < 2'000; ++i) {
    storage.emplace_back(fmt::format("debug/string/value/{}", i % 16));
  }
  values.reserve(storage.size());
  for (const auto& value : storage) {
    values.emplace_back(value);
  }

  Buffer buffer{*pool_};
  auto encoded = encodeFsst(values, buffer);
  stringBuffers_.clear();

  auto encoding =
      EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

  EXPECT_EQ(encoding->debugString(0), "FsstEncoding: 2000 rows");
}

TEST_F(FsstEncodingTest, lengthsEncodingReturnsNestedLengthsEncoding) {
  std::vector<std::string> storage;
  std::vector<std::string_view> values;
  storage.reserve(2'000);
  for (int i = 0; i < 2'000; ++i) {
    storage.emplace_back(
        fmt::format("common/prefix/for/fsst/lengths/{}", i % 16));
  }
  values.reserve(storage.size());
  for (const auto& value : storage) {
    values.emplace_back(value);
  }

  Buffer buffer{*pool_};
  auto encoded = encodeFsst(values, buffer);
  ASSERT_EQ(
      static_cast<EncodingType>(encoded[EncodingPrefix::kEncodingTypeOffset]),
      EncodingType::Fsst);

  const auto lengths = FsstEncoding::lengthsEncoding(encoded);
  auto lengthsEncoding =
      EncodingFactory().create(*pool_, lengths, createStringBufferFactory());

  EXPECT_EQ(lengthsEncoding->dataType(), DataType::Uint32);
  EXPECT_EQ(lengthsEncoding->encodingType(), EncodingType::Trivial);
  ASSERT_EQ(lengthsEncoding->rowCount(), values.size());

  std::vector<uint32_t> decodedLengths(values.size());
  lengthsEncoding->materialize(values.size(), decodedLengths.data());
  for (const auto decodedLength : decodedLengths) {
    EXPECT_GT(decodedLength, 0);
  }
}

TEST_F(FsstEncodingTest, captureNestedEncodingReturnsLengthsChildLayout) {
  std::vector<std::string> storage;
  std::vector<std::string_view> values;
  storage.reserve(2'000);
  for (int i = 0; i < 2'000; ++i) {
    storage.emplace_back(
        fmt::format("common/prefix/for/fsst/lengths/{}", i % 16));
  }
  values.reserve(storage.size());
  for (const auto& value : storage) {
    values.emplace_back(value);
  }

  Buffer buffer{*pool_};
  auto encoded = encodeFsst(values, buffer);
  ASSERT_EQ(
      static_cast<EncodingType>(encoded[EncodingPrefix::kEncodingTypeOffset]),
      EncodingType::Fsst);

  std::vector<std::optional<const EncodingLayout>> children;
  FsstEncoding::captureNestedEncoding(encoded, children);
  ASSERT_EQ(children.size(), 1);
  ASSERT_TRUE(children[0].has_value());
  EXPECT_EQ(children[0]->encodingType(), EncodingType::Trivial);
  EXPECT_EQ(children[0]->compressionType(), CompressionType::Uncompressed);
}

} // namespace
} // namespace facebook::nimble::test
