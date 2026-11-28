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

#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/EncodingSliceFactory.h"
#include "velox/dwio/nimble/encodings/MainlyIntegerEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

using namespace facebook;

namespace {

using ReadFactors = std::vector<std::pair<nimble::EncodingType, float>>;

// Pointers to the three children that MainlyIntegerEncoding::encode() hands to
// its TestValue hook. The order is (isInteger, integers, exceptions).
using ChildStreamPointers = std::tuple<
    nimble::Vector<bool>*,
    nimble::Vector<uint64_t>*,
    nimble::Vector<std::string_view>*>;

// Returns 'numRows' strings. Rows 0, 'exceptionEvery', 2 * 'exceptionEvery'
// and so on are exception rows, e.g. "word7". All other rows are integer
// rows. 'exceptionEvery' 0 means no exception rows.
std::vector<std::string> makeIntegerStrings(
    uint32_t numRows,
    uint32_t exceptionEvery) {
  std::vector<std::string> strings;
  strings.reserve(numRows);
  for (uint32_t row = 0; row < numRows; ++row) {
    if (exceptionEvery != 0 && row % exceptionEvery == 0) {
      strings.push_back(fmt::format("word{}", row));
    } else {
      strings.push_back(
          std::to_string(static_cast<int64_t>(row) * 7'919 - 5'000));
    }
  }
  return strings;
}

std::vector<std::string_view> toViews(const std::vector<std::string>& strings) {
  return {strings.begin(), strings.end()};
}

// Returns the serialized bytes of 'layout'. Tests compare two layouts by
// these bytes.
std::string serializeLayout(const nimble::EncodingLayout& layout) {
  std::string serialized(1'024, '\0');
  const auto size = layout.serialize({serialized.data(), serialized.size()});
  serialized.resize(size);
  return serialized;
}

// Reports no nulls in the read range.
class NoNullsReader {
 public:
  const uint64_t* rawNullsInReadRange() const {
    return nullptr;
  }

  bool returnReaderNulls() const {
    return false;
  }
};

// Collects the values of 'rows' that readWithVisitor() decodes.
template <typename T>
class CollectingVisitor {
 public:
  using DataType = T;
  using Extract = std::nullptr_t;

  explicit CollectingVisitor(std::vector<velox::vector_size_t> rows)
      : rows_{std::move(rows)} {}

  NoNullsReader& reader() {
    return reader_;
  }

  velox::vector_size_t currentRow() const {
    return rows_[rowIndex_];
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
  NoNullsReader reader_;
  std::vector<velox::vector_size_t> rows_;
  velox::vector_size_t rowIndex_{0};
  std::vector<T> values_;
};

} // namespace

class MainlyIntegerEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::common::testutil::TestValue::enable();
  }

  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  // Keeps every decoded string alive for the rest of the test.
  std::function<void*(uint32_t)> stringBufferFactory() {
    return [this](uint32_t size) -> void* {
      return stringBuffers_.emplace_back(size, '\0').data();
    };
  }

  // Encodes 'values' as MainlyInteger. Each child uses the Trivial encoding.
  std::string_view encodeMainlyInteger(
      const std::vector<std::string_view>& values,
      const nimble::Encoding::Options& options = {}) {
    return nimble::test::Encoder<nimble::MainlyIntegerEncoding>::encode(
        *buffer_,
        nimble::Vector<std::string_view>{
            pool_.get(), values.begin(), values.end()},
        nimble::CompressionType::Uncompressed,
        options);
  }

  std::string_view encodeStrings(
      const std::vector<std::string>& strings,
      const nimble::Encoding::Options& options = {}) {
    return encodeMainlyInteger(toViews(strings), options);
  }

  // Encodes 'values' with a policy that may pick from 'readFactors'.
  template <typename T>
  std::string_view encodeWithPolicy(
      const std::vector<T>& values,
      const ReadFactors& readFactors) {
    auto policy = std::make_unique<nimble::ManualEncodingSelectionPolicy<T>>(
        readFactors,
        /*compressionOptions=*/std::nullopt,
        /*identifier=*/std::nullopt);
    return nimble::EncodingFactory::encode<T>(
        std::move(policy), std::span<const T>{values}, *buffer_);
  }

  // Decodes every row of 'encoded'. Then calls reset(), skips the first third
  // of the rows and decodes the rest. Returns both results.
  template <typename T>
  std::pair<std::vector<T>, std::vector<T>> decodeTwice(
      std::string_view encoded,
      const nimble::Encoding::Options& options = {}) {
    auto encoding = nimble::EncodingFactory{options}.create(
        *pool_, encoded, stringBufferFactory());
    EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::MainlyInteger);
    const auto numRows = encoding->rowCount();
    std::vector<T> all(numRows);
    encoding->materialize(numRows, all.data());
    encoding->reset();
    const uint32_t numSkipped = numRows / 3;
    encoding->skip(numSkipped);
    std::vector<T> rest(numRows - numSkipped);
    encoding->materialize(numRows - numSkipped, rest.data());
    return {std::move(all), std::move(rest)};
  }

  void checkStrings(
      std::string_view encoded,
      const std::vector<std::string>& expected,
      const nimble::Encoding::Options& options = {}) {
    const auto [all, rest] = decodeTwice<std::string_view>(encoded, options);
    EXPECT_THAT(all, testing::ElementsAreArray(expected));
    EXPECT_THAT(
        rest,
        testing::ElementsAreArray(
            expected.begin() + expected.size() / 3, expected.end()));
  }

  // Returns the byte sizes of the is-integer stream and the integers child of
  // 'encoded'.
  static std::pair<uint32_t, uint32_t> childSizes(std::string_view encoded) {
    std::string_view body = encoded.substr(
        nimble::EncodingPrefix::prefixSize(encoded, /*useVarint=*/false));
    const uint32_t isIntegerBytes = nimble::encoding::readVarint32(body);
    body.remove_prefix(isIntegerBytes);
    return {isIntegerBytes, nimble::encoding::readVarint32(body)};
  }

  // Expects 'actual' to equal 'expected'.
  template <typename T>
  static void expectValues(
      const std::vector<T>& actual,
      const std::vector<T>& expected) {
    EXPECT_THAT(actual, testing::ElementsAreArray(expected));
  }

  // Byte offsets in a MainlyInteger stream, from its documented layout.
  struct ChildOffsets {
    // Where the is-integer size starts, right after the prefix.
    size_t isIntegerSize;
    // Where the is-integer stream starts.
    size_t isInteger;
    // Where the integer size starts.
    size_t integerSize;
    // Where the integers start.
    size_t integers;
    // Where the exceptions start.
    size_t exceptions;
  };

  static ChildOffsets childOffsets(std::string_view encoded) {
    const auto [isIntegerBytes, integerBytes] = childSizes(encoded);
    ChildOffsets offsets{};
    offsets.isIntegerSize =
        nimble::EncodingPrefix::prefixSize(encoded, /*useVarint=*/false);
    offsets.isInteger =
        offsets.isIntegerSize + nimble::varint::varintSize(isIntegerBytes);
    offsets.integerSize = offsets.isInteger + isIntegerBytes;
    offsets.integers =
        offsets.integerSize + nimble::varint::varintSize(integerBytes);
    offsets.exceptions = offsets.integers + integerBytes;
    return offsets;
  }

  // Returns 'encoded' with the varint at 'offset' replaced by 'value'.
  static std::string
  replaceVarint(std::string_view encoded, size_t offset, uint32_t value) {
    std::string_view rest = encoded.substr(offset);
    nimble::encoding::readVarint32(rest);
    char bytes[nimble::varint::maxVarintSizeForBitWidth(32)];
    char* end = bytes;
    nimble::encoding::writeVarintUint32(value, end);
    std::string replaced{encoded.substr(0, offset)};
    replaced.append(bytes, end - bytes);
    replaced.append(rest);
    return replaced;
  }

  // Decodes the integers child of 'encoded' and returns its values.
  std::vector<uint64_t> storedIntegers(std::string_view encoded) {
    const auto offsets = childOffsets(encoded);
    const std::string_view integerChild{
        encoded.data() + offsets.integers,
        offsets.exceptions - offsets.integers};
    auto integers = nimble::EncodingFactory().create(
        *pool_, integerChild, stringBufferFactory());
    std::vector<uint64_t> values(integers->rowCount());
    integers->materialize(integers->rowCount(), values.data());
    return values;
  }

  // Encodes 'values'. Before the children are encoded, 'mutate' can change them
  // through the TestValue hook, e.g. to build a corrupted stream. TestValue
  // hooks run only in debug builds.
  std::string_view encodeWithChildStreams(
      const std::vector<std::string_view>& values,
      std::function<void(ChildStreamPointers*)> mutate) {
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::MainlyIntegerEncoding::encode", std::move(mutate));
    return encodeMainlyInteger(values);
  }

  // Decodes each shape with one encoding object. The first shape creates the
  // encoding, and each later shape rebinds it. Each shape is read in full,
  // then reset and half read again. So the next rebind follows a partial
  // read.
  void testRebinds(const std::vector<std::vector<std::string_view>>& shapes) {
    std::unique_ptr<nimble::Encoding> encoding;
    for (const auto& values : shapes) {
      const auto encoded = encodeMainlyInteger(values);
      if (encoding == nullptr) {
        encoding = nimble::EncodingFactory().create(
            *pool_, encoded, stringBufferFactory());
      } else {
        ASSERT_TRUE(encoding->rebind(encoded, stringBufferFactory()));
      }
      ASSERT_EQ(encoding->rowCount(), values.size());
      std::vector<std::string_view> output(values.size());
      encoding->materialize(encoding->rowCount(), output.data());
      expectValues(output, values);
      encoding->reset();
      encoding->materialize(encoding->rowCount() / 2, output.data());
    }
  }

  // Checks reads that mix readWithVisitor() with skip() and materialize().
  // 'values' needs at least 3'000 rows. readWithVisitor() decodes rows before
  // the caller asks for them. skip() and materialize() must use those rows
  // before they read the children. A rebind to 'rebindValues' must drop them.
  void testReadWithVisitor(
      const std::vector<std::string_view>& values,
      const std::vector<std::string_view>& rebindValues) {
    ASSERT_GE(values.size(), 3'000);
    nimble::MainlyIntegerEncoding encoding{
        *pool_, encodeMainlyInteger(values), stringBufferFactory()};
    const std::vector<velox::vector_size_t> rows{0, 7, 1'023, 1'024, 1'500};
    CollectingVisitor<std::string_view> visitor{rows};
    nimble::ReadWithVisitorParams params;
    params.numScanned = 0;
    encoding.readWithVisitor(visitor, params);
    std::vector<std::string_view> expected;
    expected.reserve(rows.size());
    for (const auto row : rows) {
      expected.push_back(values[row]);
    }
    expectValues(visitor.values(), expected);

    // Rows 1'501 to 2'047 are decoded ahead.
    encoding.skip(100);
    std::vector<std::string_view> rest(values.size() - 1'601);
    encoding.materialize(static_cast<uint32_t>(rest.size()), rest.data());
    expectValues(
        rest,
        std::vector<std::string_view>(values.begin() + 1'601, values.end()));

    encoding.reset();
    CollectingVisitor<std::string_view> firstRow{{0}};
    encoding.readWithVisitor(firstRow, params);
    ASSERT_TRUE(encoding.rebind(
        encodeMainlyInteger(rebindValues), stringBufferFactory()));
    std::vector<std::string_view> rebound(rebindValues.size());
    encoding.materialize(static_cast<uint32_t>(rebound.size()), rebound.data());
    expectValues(rebound, rebindValues);
  }

  // Creates an encoding of 'values' and reads rows 0 and 7 with
  // readWithVisitor(). That decodes the first 1'024 rows ahead, so 8 rows are
  // read and the read-ahead rows are not empty. 'values' needs more than
  // 1'024 rows.
  std::unique_ptr<nimble::MainlyIntegerEncoding> createAfterReadAhead(
      const std::vector<std::string_view>& values) {
    auto encoding = std::make_unique<nimble::MainlyIntegerEncoding>(
        *pool_, encodeMainlyInteger(values), stringBufferFactory());
    CollectingVisitor<std::string_view> visitor{{0, 7}};
    nimble::ReadWithVisitorParams params;
    params.numScanned = 0;
    encoding->readWithVisitor(visitor, params);
    expectValues(
        visitor.values(), std::vector<std::string_view>{values[0], values[7]});
    return encoding;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::deque<std::string> stringBuffers_;
};

TEST_F(MainlyIntegerEncodingTest, integerStringsAcceptOnlyCanonicalSpellings) {
  for (const auto& [accepted, integer] :
       std::vector<std::pair<std::string_view, int64_t>>{
           {"0", 0},
           {"-42", -42},
           {"9223372036854775807", std::numeric_limits<int64_t>::max()},
           {"-9223372036854775808", std::numeric_limits<int64_t>::min()},
       }) {
    SCOPED_TRACE(accepted);
    const auto encoded = encodeMainlyInteger({accepted});
    EXPECT_EQ(childSizes(encoded).first, 0);
    EXPECT_THAT(
        storedIntegers(encoded),
        testing::ElementsAre(static_cast<uint64_t>(integer)));
  }
  for (const std::string_view rejected :
       {"007",
        "+1",
        "-0",
        "",
        "1e3",
        " 1",
        "1 ",
        "9223372036854775808",
        "-9223372036854775809",
        "-",
        "12a",
        "0x10"}) {
    SCOPED_TRACE(fmt::format("'{}'", rejected));
    // A lone exception row stores only the exceptions.
    const auto encoded = encodeMainlyInteger({rejected});
    EXPECT_EQ(childSizes(encoded), std::make_pair(0u, 0u));
    EXPECT_LT(childOffsets(encoded).exceptions, encoded.size());
  }
}

TEST_F(MainlyIntegerEncodingTest, integerStringsRoundTrip) {
  const std::vector<std::string> edgeStrings{
      "0",
      "7",
      "-7",
      "9223372036854775807",
      "-9223372036854775808",
      "007",
      "+1",
      "-0",
      "",
      "1e3",
      " 1",
      "9223372036854775808",
      "-",
      "12a",
  };
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    checkStrings(encodeStrings(edgeStrings, options), edgeStrings, options);
  }
  const auto mixed =
      makeIntegerStrings(/*numRows=*/3'000, /*exceptionEvery=*/37);
  checkStrings(encodeStrings(mixed), mixed);
}

TEST_F(
    MainlyIntegerEncodingTest,
    integerStringsWithoutExceptionsStoreNoIsInteger) {
  const auto strings =
      makeIntegerStrings(/*numRows=*/500, /*exceptionEvery=*/0);
  const auto encoded = encodeStrings(strings);
  const auto [isIntegerBytes, integerBytes] = childSizes(encoded);
  EXPECT_EQ(isIntegerBytes, 0);
  EXPECT_GT(integerBytes, 0);
  checkStrings(encoded, strings);
}

TEST_F(
    MainlyIntegerEncodingTest,
    integerStringsWithoutIntegersStoreOnlyExceptions) {
  const std::vector<std::string> strings{"a", "", "007", "b"};
  const auto encoded = encodeStrings(strings);
  const auto [isIntegerBytes, integerBytes] = childSizes(encoded);
  EXPECT_EQ(isIntegerBytes, 0);
  EXPECT_EQ(integerBytes, 0);
  checkStrings(encoded, strings);
}

TEST_F(MainlyIntegerEncodingTest, encodesEmptyInput) {
  const auto encoded = encodeMainlyInteger({});
  const auto [isIntegerBytes, integerBytes] = childSizes(encoded);
  EXPECT_EQ(isIntegerBytes, 0);
  EXPECT_EQ(integerBytes, 0);
  EXPECT_EQ(childOffsets(encoded).exceptions, encoded.size());
  auto encoding =
      nimble::EncodingFactory().create(*pool_, encoded, stringBufferFactory());
  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::MainlyInteger);
  EXPECT_EQ(encoding->rowCount(), 0);
  encoding->skip(0);
  encoding->materialize(0, nullptr);
  NIMBLE_ASSERT_THROW(
      encoding->skip(1), "Skipping past the end of the MainlyInteger encoding");
}

TEST_F(MainlyIntegerEncodingTest, integerStringsSelectedWhenEnabled) {
  const auto strings =
      makeIntegerStrings(/*numRows=*/2'000, /*exceptionEvery=*/50);
  const ReadFactors withoutMainlyInteger{
      {nimble::EncodingType::Trivial, 1.0},
      {nimble::EncodingType::Dictionary, 1.0},
  };
  ReadFactors withMainlyInteger = withoutMainlyInteger;
  withMainlyInteger.emplace_back(nimble::EncodingType::MainlyInteger, 1.0);
  const auto views = toViews(strings);
  const auto enabled = encodeWithPolicy(views, withMainlyInteger);
  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(enabled),
      nimble::EncodingType::MainlyInteger);
  checkStrings(enabled, strings);
  EXPECT_NE(
      nimble::EncodingPrefix::encodingType(
          encodeWithPolicy(views, withoutMainlyInteger)),
      nimble::EncodingType::MainlyInteger);
}

TEST_F(MainlyIntegerEncodingTest, integerStringsNotSelectedForOtherStrings) {
  const auto strings =
      makeIntegerStrings(/*numRows=*/2'000, /*exceptionEvery=*/2);
  EXPECT_EQ(
      nimble::MainlyIntegerEncoding::estimateSize(
          toViews(strings), nimble::Encoding::Options{}),
      std::nullopt);
  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(encodeWithPolicy(
          toViews(strings),
          {{nimble::EncodingType::Trivial, 1.0},
           {nimble::EncodingType::MainlyInteger, 1.0}})),
      nimble::EncodingType::Trivial);
}

TEST_F(MainlyIntegerEncodingTest, integerStringsReadWithVisitor) {
  const auto strings =
      makeIntegerStrings(/*numRows=*/3'000, /*exceptionEvery=*/13);
  const std::vector<std::string> rebindStrings{"5", "x", "-6"};
  testReadWithVisitor(toViews(strings), toViews(rebindStrings));
}

TEST_F(
    MainlyIntegerEncodingTest,
    integerStringsWithoutIntegersReadWithVisitor) {
  const auto words =
      makeIntegerStrings(/*numRows=*/3'000, /*exceptionEvery=*/1);
  const auto rebindStrings =
      makeIntegerStrings(/*numRows=*/40, /*exceptionEvery=*/1);
  testReadWithVisitor(toViews(words), toViews(rebindStrings));
}

TEST_F(MainlyIntegerEncodingTest, integerStringsRebindAcrossShapes) {
  const auto mixed = makeIntegerStrings(/*numRows=*/200, /*exceptionEvery=*/9);
  const auto allIntegers =
      makeIntegerStrings(/*numRows=*/300, /*exceptionEvery=*/0);
  const std::vector<std::string> noIntegers{"x", "y"};
  testRebinds({
      toViews(allIntegers),
      toViews(mixed),
      toViews(noIntegers),
      toViews(allIntegers),
      toViews(mixed),
  });
}

TEST_F(MainlyIntegerEncodingTest, capturesNestedLayouts) {
  const auto strings =
      makeIntegerStrings(/*numRows=*/400, /*exceptionEvery=*/0);
  const auto captured = nimble::EncodingLayoutCapture::capture(
      encodeStrings(strings), nimble::Encoding::Options{});
  EXPECT_EQ(captured.encodingType(), nimble::EncodingType::MainlyInteger);
  ASSERT_EQ(captured.childrenCount(), 3);
  EXPECT_FALSE(
      captured.child(nimble::EncodingIdentifiers::MainlyInteger::IsInteger)
          .has_value());
  EXPECT_TRUE(
      captured.child(nimble::EncodingIdentifiers::MainlyInteger::Integers)
          .has_value());
  EXPECT_FALSE(
      captured.child(nimble::EncodingIdentifiers::MainlyInteger::Exceptions)
          .has_value());
}

TEST_F(MainlyIntegerEncodingTest, capturesOnlyExceptionsLayout) {
  const std::vector<std::string> strings{"a", "b", "007"};
  const auto captured = nimble::EncodingLayoutCapture::capture(
      encodeStrings(strings), nimble::Encoding::Options{});
  EXPECT_EQ(captured.encodingType(), nimble::EncodingType::MainlyInteger);
  ASSERT_EQ(captured.childrenCount(), 3);
  EXPECT_FALSE(
      captured.child(nimble::EncodingIdentifiers::MainlyInteger::IsInteger)
          .has_value());
  EXPECT_FALSE(
      captured.child(nimble::EncodingIdentifiers::MainlyInteger::Integers)
          .has_value());
  EXPECT_TRUE(
      captured.child(nimble::EncodingIdentifiers::MainlyInteger::Exceptions)
          .has_value());
}

TEST_F(MainlyIntegerEncodingTest, replaysCapturedLayout) {
  const auto strings =
      makeIntegerStrings(/*numRows=*/2'000, /*exceptionEvery=*/50);
  const auto views = toViews(strings);
  const auto encoded = encodeWithPolicy(
      views,
      {{nimble::EncodingType::Trivial, 1.0},
       {nimble::EncodingType::MainlyInteger, 1.0}});
  const auto captured = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});
  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::MainlyInteger);

  const nimble::EncodingSelectionPolicyCreator policyCreator =
      [](nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return nimble::ManualEncodingSelectionPolicyFactory{}.createPolicy(
        dataType);
  };
  const auto replayed = nimble::EncodingFactory::encode<std::string_view>(
      std::make_unique<
          nimble::ReplayedEncodingSelectionPolicy<std::string_view>>(
          captured, /*compressionOptions=*/std::nullopt, policyCreator),
      std::span<const std::string_view>{views},
      *buffer_);

  EXPECT_EQ(
      serializeLayout(
          nimble::EncodingLayoutCapture::capture(
              replayed, nimble::Encoding::Options{})),
      serializeLayout(captured));
  checkStrings(replayed, strings);
}

TEST_F(MainlyIntegerEncodingTest, replaysCapturedLayoutAcrossShapes) {
  // A layout captured from one shape is replayed on data of every shape. A
  // missing child layout falls back to normal selection. A child layout for
  // a child the data does not need is unused.
  const std::vector<std::pair<std::string, std::vector<std::string>>> shapes{
      {"mixed", makeIntegerStrings(/*numRows=*/600, /*exceptionEvery=*/7)},
      {"allIntegers",
       makeIntegerStrings(/*numRows=*/600, /*exceptionEvery=*/0)},
      {"allExceptions",
       makeIntegerStrings(/*numRows=*/600, /*exceptionEvery=*/1)},
      {"empty", {}},
  };
  const nimble::EncodingSelectionPolicyCreator policyCreator =
      [](nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return nimble::ManualEncodingSelectionPolicyFactory{}.createPolicy(
        dataType);
  };
  for (const auto& [sourceName, source] : shapes) {
    if (source.empty()) {
      continue;
    }
    const auto captured = nimble::EncodingLayoutCapture::capture(
        encodeStrings(source), nimble::Encoding::Options{});
    for (const auto& [targetName, target] : shapes) {
      SCOPED_TRACE(fmt::format("{} -> {}", sourceName, targetName));
      const auto views = toViews(target);
      const auto replayed = nimble::EncodingFactory::encode<std::string_view>(
          std::make_unique<
              nimble::ReplayedEncodingSelectionPolicy<std::string_view>>(
              captured, /*compressionOptions=*/std::nullopt, policyCreator),
          std::span<const std::string_view>{views},
          *buffer_);
      ASSERT_EQ(
          nimble::EncodingPrefix::encodingType(replayed),
          nimble::EncodingType::MainlyInteger);
      checkStrings(replayed, target);
    }
  }
}

TEST_F(MainlyIntegerEncodingTest, slicesByMaterializing) {
  const auto strings = makeIntegerStrings(/*numRows=*/90, /*exceptionEvery=*/7);
  nimble::Buffer sliceBuffer{*pool_};
  const auto sliced = nimble::EncodingSliceFactory::slice(
      encodeStrings(strings), /*offset=*/10, /*length=*/50, sliceBuffer, {});
  auto encoding =
      nimble::EncodingFactory().create(*pool_, sliced, stringBufferFactory());
  std::vector<std::string_view> output(50);
  encoding->materialize(50, output.data());
  EXPECT_THAT(
      output,
      testing::ElementsAreArray(strings.begin() + 10, strings.begin() + 60));
}

TEST_F(MainlyIntegerEncodingTest, viewIsNotSupported) {
  const auto encoded = encodeStrings({"1", "2"});
  NIMBLE_ASSERT_USER_THROW(
      nimble::createEncodingView(encoded, pool_.get(), {}),
      "does not support EncodingView");
}

DEBUG_ONLY_TEST_F(
    MainlyIntegerEncodingTest,
    rejectsMoreIntegerRowsThanIntegers) {
  // The is-integer stream marks two integer rows, but the stream stores one
  // integer. An extra exception keeps the children covering every row.
  const auto encoded =
      encodeWithChildStreams({"1", "2", "a"}, [](ChildStreamPointers* streams) {
        std::get<1>(*streams)->resize(1);
        std::get<2>(*streams)->push_back("b");
      });
  {
    auto encoding = nimble::EncodingFactory().create(
        *pool_, encoded, stringBufferFactory());
    std::vector<std::string_view> output(3);
    NIMBLE_ASSERT_FILE_THROW(
        encoding->materialize(3, output.data()),
        "MainlyInteger is-integer stream marks more rows than the child stores: integers");
  }
  {
    auto encoding = nimble::EncodingFactory().create(
        *pool_, encoded, stringBufferFactory());
    NIMBLE_ASSERT_FILE_THROW(
        encoding->skip(3),
        "MainlyInteger is-integer stream marks more rows than the child stores: integers");
  }
}

DEBUG_ONLY_TEST_F(MainlyIntegerEncodingTest, rejectsIsIntegerCountMismatch) {
  // One integer and one exception, but the is-integer stream marks both rows.
  {
    auto encoding = nimble::EncodingFactory().create(
        *pool_,
        encodeWithChildStreams(
            {"7", "a"},
            [](ChildStreamPointers* streams) {
              (*std::get<0>(*streams))[1] = true;
            }),
        stringBufferFactory());
    std::vector<std::string_view> output(2);
    NIMBLE_ASSERT_FILE_THROW(
        encoding->materialize(2, output.data()),
        "MainlyInteger is-integer stream marks more rows than the child stores: integers");
  }
  // The same children, but the is-integer stream marks neither row.
  {
    auto encoding = nimble::EncodingFactory().create(
        *pool_,
        encodeWithChildStreams(
            {"7", "a"},
            [](ChildStreamPointers* streams) {
              (*std::get<0>(*streams))[0] = false;
            }),
        stringBufferFactory());
    NIMBLE_ASSERT_FILE_THROW(
        encoding->skip(2),
        "MainlyInteger is-integer stream marks more rows than the child stores: "
        "exceptions");
  }
}

DEBUG_ONLY_TEST_F(MainlyIntegerEncodingTest, rejectsIsIntegerRowCountMismatch) {
  // The is-integer stream has one row for a stream of two rows.
  const auto encoded = encodeWithChildStreams(
      {"7", "a"},
      [](ChildStreamPointers* streams) { std::get<0>(*streams)->resize(1); });
  NIMBLE_ASSERT_FILE_THROW(
      nimble::EncodingFactory().create(*pool_, encoded, stringBufferFactory()),
      "MainlyInteger is-integer stream must have one row per stream row");
}

DEBUG_ONLY_TEST_F(
    MainlyIntegerEncodingTest,
    rejectsChildrenNotCoveringEveryRow) {
  // Three rows, but no child holds any of them.
  const auto encoded =
      encodeWithChildStreams({"1", "2", "x"}, [](ChildStreamPointers* streams) {
        std::get<1>(*streams)->clear();
        std::get<2>(*streams)->clear();
      });
  NIMBLE_ASSERT_FILE_THROW(
      nimble::EncodingFactory().create(*pool_, encoded, stringBufferFactory()),
      "MainlyInteger children must cover every row");
}

TEST_F(MainlyIntegerEncodingTest, rejectsSkipPastTheEnd) {
  const auto strings = makeIntegerStrings(3'000, 7);
  const auto values = toViews(strings);
  {
    nimble::MainlyIntegerEncoding encoding{
        *pool_, encodeMainlyInteger(values), stringBufferFactory()};
    NIMBLE_ASSERT_THROW(
        encoding.skip(3'001),
        "Skipping past the end of the MainlyInteger encoding");
  }
  {
    // 8 rows are read, so 2'992 rows remain.
    auto encoding = createAfterReadAhead(values);
    NIMBLE_ASSERT_THROW(
        encoding->skip(2'993),
        "Skipping past the end of the MainlyInteger encoding");
  }
}

TEST_F(MainlyIntegerEncodingTest, rejectsMaterializePastTheEnd) {
  const auto strings = makeIntegerStrings(3'000, 7);
  const auto values = toViews(strings);
  std::vector<std::string_view> output(3'001);
  {
    nimble::MainlyIntegerEncoding encoding{
        *pool_, encodeMainlyInteger(values), stringBufferFactory()};
    NIMBLE_ASSERT_THROW(
        encoding.materialize(3'001, output.data()),
        "Materializing past the end of the MainlyInteger encoding");
  }
  {
    // 8 rows are read, so 2'992 rows remain.
    auto encoding = createAfterReadAhead(values);
    NIMBLE_ASSERT_THROW(
        encoding->materialize(2'993, output.data()),
        "Materializing past the end of the MainlyInteger encoding");
  }
}

TEST_F(MainlyIntegerEncodingTest, readsUpToTheExactEnd) {
  const auto strings = makeIntegerStrings(3'000, 7);
  const auto values = toViews(strings);
  {
    nimble::MainlyIntegerEncoding encoding{
        *pool_, encodeMainlyInteger(values), stringBufferFactory()};
    std::vector<std::string_view> output(3'000);
    encoding.materialize(3'000, output.data());
    expectValues(output, values);
    encoding.reset();
    encoding.skip(3'000);
    NIMBLE_ASSERT_THROW(
        encoding.skip(1),
        "Skipping past the end of the MainlyInteger encoding");
  }
  {
    auto encoding = createAfterReadAhead(values);
    std::vector<std::string_view> output(2'992);
    encoding->materialize(2'992, output.data());
    expectValues(
        output,
        std::vector<std::string_view>(values.begin() + 8, values.end()));
    NIMBLE_ASSERT_THROW(
        encoding->materialize(1, output.data()),
        "Materializing past the end of the MainlyInteger encoding");
  }
  {
    auto encoding = createAfterReadAhead(values);
    encoding->skip(2'992);
    NIMBLE_ASSERT_THROW(
        encoding->skip(1),
        "Skipping past the end of the MainlyInteger encoding");
  }
}

TEST_F(MainlyIntegerEncodingTest, rejectsIsIntegerWithoutExceptions) {
  // A stream cut right after its integer child keeps the is-integer stream
  // but loses the exceptions.
  const auto encoded = encodeMainlyInteger({"1", "x"});
  const std::string withoutExceptions{
      encoded.substr(0, childOffsets(encoded).exceptions)};
  NIMBLE_ASSERT_FILE_THROW(
      nimble::EncodingFactory().create(
          *pool_, withoutExceptions, stringBufferFactory()),
      "MainlyInteger stores the is-integer stream exactly when it stores both "
      "integers and exceptions");
}

TEST_F(MainlyIntegerEncodingTest, rejectsIsIntegerWithoutIntegers) {
  // The integers are cut out and their size set to 0; the is-integer stream
  // and the exceptions remain.
  const auto encoded = encodeMainlyInteger({"1", "x"});
  const auto offsets = childOffsets(encoded);
  std::string withoutIntegers{encoded.substr(0, offsets.integers)};
  withoutIntegers.append(encoded.substr(offsets.exceptions));
  withoutIntegers = replaceVarint(withoutIntegers, offsets.integerSize, 0);
  NIMBLE_ASSERT_FILE_THROW(
      nimble::EncodingFactory().create(
          *pool_, withoutIntegers, stringBufferFactory()),
      "MainlyInteger stores the is-integer stream exactly when it stores both "
      "integers and exceptions");
}

TEST_F(MainlyIntegerEncodingTest, rejectsExceptionsWithoutIsInteger) {
  // The is-integer stream is cut out and its size set to 0; the integers and
  // the exceptions remain.
  const auto encoded = encodeMainlyInteger({"1", "x"});
  const auto offsets = childOffsets(encoded);
  std::string withoutIsInteger{encoded.substr(0, offsets.isInteger)};
  withoutIsInteger.append(encoded.substr(offsets.integerSize));
  withoutIsInteger = replaceVarint(withoutIsInteger, offsets.isIntegerSize, 0);
  NIMBLE_ASSERT_FILE_THROW(
      nimble::EncodingFactory().create(
          *pool_, withoutIsInteger, stringBufferFactory()),
      "MainlyInteger stores the is-integer stream exactly when it stores both "
      "integers and exceptions");
}

TEST_F(MainlyIntegerEncodingTest, rejectsTruncatedStreams) {
  const auto encoded = encodeMainlyInteger({"1", "x"});
  const auto offsets = childOffsets(encoded);
  for (const auto& [length, error] :
       std::vector<std::pair<size_t, std::string_view>>{
           // Before the is-integer size.
           {offsets.isIntegerSize, "Truncated uint32 varint."},
           // Inside the is-integer stream.
           {offsets.integerSize - 1, "Truncated length-prefixed payload."},
           // Before the integer size.
           {offsets.integerSize, "Truncated uint32 varint."},
           // Inside the integers.
           {offsets.exceptions - 1, "Truncated length-prefixed payload."},
       }) {
    SCOPED_TRACE(length);
    const std::string truncated{encoded.substr(0, length)};
    NIMBLE_ASSERT_THROW(
        nimble::EncodingFactory().create(
            *pool_, truncated, stringBufferFactory()),
        error);
  }
}

TEST_F(MainlyIntegerEncodingTest, rejectsChildSizesPastTheEnd) {
  const auto encoded = encodeMainlyInteger({"1", "x"});
  const auto offsets = childOffsets(encoded);
  for (const size_t sizeOffset : {offsets.isIntegerSize, offsets.integerSize}) {
    SCOPED_TRACE(sizeOffset);
    const std::string patched = replaceVarint(
        encoded, sizeOffset, static_cast<uint32_t>(encoded.size()));
    NIMBLE_ASSERT_THROW(
        nimble::EncodingFactory().create(
            *pool_, patched, stringBufferFactory()),
        "Truncated length-prefixed payload.");
  }
}
