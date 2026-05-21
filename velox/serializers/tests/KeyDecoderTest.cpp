/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#include "velox/serializers/KeyDecoder.h"

#include <cstring>
#include <limits>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::serializer::test {
namespace {

class KeyDecoderTest : public velox::exec::test::OperatorTestBase {
 protected:
  template <typename T>
  static T& checkedReference(const std::unique_ptr<T>& pointer) {
    VELOX_CHECK_NOT_NULL(pointer.get());
    // @lint-ignore NULLSAFECLANG nullable-dereference
    return *pointer;
  }

  RowVectorPtr encodeAndDecode(
      const RowVectorPtr& input,
      const std::vector<std::string>& keyColumns,
      const std::vector<velox::core::SortOrder>& sortOrders) {
    auto encoder = KeyEncoder::create(
        keyColumns, asRowType(input->type()), sortOrders, pool_.get());
    auto decoder = KeyDecoder::create(
        keyColumns, asRowType(input->type()), sortOrders, pool_.get());
    auto& keyEncoder = checkedReference(encoder);
    auto& keyDecoder = checkedReference(decoder);

    std::vector<char> buffer;
    std::vector<std::string_view> encodedKeys;
    keyEncoder.encode(input, encodedKeys, [&buffer](size_t size) -> void* {
      buffer.resize(size);
      return buffer.data();
    });

    return keyDecoder.decode(std::span<const std::string_view>(encodedKeys));
  }

  std::vector<std::string> encodeToStrings(
      const RowVectorPtr& input,
      const std::vector<std::string>& keyColumns,
      const std::vector<velox::core::SortOrder>& sortOrders) {
    auto encoder = KeyEncoder::create(
        keyColumns, asRowType(input->type()), sortOrders, pool_.get());
    auto& keyEncoder = checkedReference(encoder);

    std::vector<char> buffer;
    std::vector<std::string_view> encodedViews;
    keyEncoder.encode(input, encodedViews, [&buffer](size_t size) -> void* {
      buffer.resize(size);
      return buffer.data();
    });

    std::vector<std::string> encodedKeys;
    encodedKeys.reserve(encodedViews.size());
    for (const auto encodedView : encodedViews) {
      encodedKeys.emplace_back(encodedView.data(), encodedView.size());
    }
    return encodedKeys;
  }

  RowVectorPtr projectKeyColumns(
      const RowVectorPtr& input,
      const std::vector<std::string>& keyColumns) {
    const auto inputType = asRowType(input->type());

    std::vector<TypePtr> childTypes;
    std::vector<VectorPtr> children;
    childTypes.reserve(keyColumns.size());
    children.reserve(keyColumns.size());

    for (const auto& keyColumn : keyColumns) {
      const auto channel = inputType->getChildIdx(keyColumn);
      childTypes.push_back(inputType->childAt(channel));
      children.push_back(input->childAt(channel));
    }

    return std::make_shared<RowVector>(
        pool_.get(),
        ROW(keyColumns, childTypes),
        nullptr,
        input->size(),
        children);
  }

  std::vector<velox::core::SortOrder> repeatedSortOrders(
      size_t count,
      const velox::core::SortOrder& sortOrder) {
    return std::vector<velox::core::SortOrder>(count, sortOrder);
  }

  static StringView stringView(std::string_view value) {
    VELOX_CHECK_LE(
        value.size(),
        static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "Test string exceeds StringView maximum length: {}",
        value.size());
    return StringView(value.data(), static_cast<int32_t>(value.size()));
  }

  static double bitsToDouble(uint64_t bits) {
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  static float bitsToFloat(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  static uint64_t doubleToBits(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }

  static uint32_t floatToBits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }
};

TEST_F(KeyDecoderTest, roundTripAllSupportedTypesWithoutNulls) {
  const std::string varchar0("alpha");
  const std::string varchar1({char(0), char(1), 'b', 'e', 't', 'a'});
  const std::string varchar2("omega");

  const std::string binary0({char(0), char(1), char(2)});
  const std::string binary1({char(0xff), char(0), char(1)});
  const std::string binary2("bin");

  auto input = makeRowVector(
      {makeFlatVector<int64_t>({-9, 0, 99}),
       makeFlatVector<int32_t>({-1234, 0, 5678}),
       makeFlatVector<int16_t>({-12, 0, 12}),
       makeFlatVector<int8_t>({-5, 0, 5}),
       makeFlatVector<double>({-12.5, 0.5, 42.25}),
       makeFlatVector<float>({-3.5f, 0.25f, 7.75f}),
       makeFlatVector<bool>({true, false, true}),
       makeFlatVector<StringView>(
           {stringView(varchar0), stringView(varchar1), stringView(varchar2)},
           VARCHAR()),
       makeFlatVector<StringView>(
           {stringView(binary0), stringView(binary1), stringView(binary2)},
           VARBINARY()),
       makeFlatVector<Timestamp>(
           {Timestamp(-100, 1), Timestamp(0, 0), Timestamp(123, 999'999'999)}),
       makeFlatVector<int32_t>({-1, 0, 365}, DATE())});

  const std::vector<std::string> keyColumns = {
      "c8", "c0", "c10", "c7", "c3", "c9", "c1", "c6", "c2", "c4", "c5"};

  const auto expected = projectKeyColumns(input, keyColumns);
  for (const auto& sortOrder :
       {velox::core::kAscNullsFirst,
        velox::core::kAscNullsLast,
        velox::core::kDescNullsFirst,
        velox::core::kDescNullsLast}) {
    SCOPED_TRACE(
        fmt::format(
            "sortOrder={}{}",
            sortOrder.isAscending() ? "ASC" : "DESC",
            sortOrder.isNullsFirst() ? "_NULLS_FIRST" : "_NULLS_LAST"));
    const auto decoded = encodeAndDecode(
        input, keyColumns, repeatedSortOrders(keyColumns.size(), sortOrder));
    velox::test::assertEqualVectors(expected, decoded);
  }
}

TEST_F(KeyDecoderTest, roundTripAllSupportedTypesWithNulls) {
  const std::string varchar0("alpha");
  const std::string varchar2({char(0), char(1), 't', 'a', 'i', 'l'});

  const std::string binary0({char(0), char(1)});
  const std::string binary2({char(0xfe), char(0xff)});

  auto input = makeRowVector(
      {makeNullableFlatVector<int64_t>({-9, std::nullopt, 99}),
       makeNullableFlatVector<int32_t>({-1234, std::nullopt, 5678}),
       makeNullableFlatVector<int16_t>({-12, std::nullopt, 12}),
       makeNullableFlatVector<int8_t>({-5, std::nullopt, 5}),
       makeNullableFlatVector<double>({-12.5, std::nullopt, 42.25}),
       makeNullableFlatVector<float>({-3.5f, std::nullopt, 7.75f}),
       makeNullableFlatVector<bool>({true, std::nullopt, false}),
       makeNullableFlatVector<StringView>(
           {stringView(varchar0), std::nullopt, stringView(varchar2)},
           VARCHAR()),
       makeNullableFlatVector<StringView>(
           {stringView(binary0), std::nullopt, stringView(binary2)},
           VARBINARY()),
       makeNullableFlatVector<Timestamp>(
           {Timestamp(-100, 1), std::nullopt, Timestamp(123, 999'999'999)}),
       makeNullableFlatVector<int32_t>({-1, std::nullopt, 365}, DATE())});

  const std::vector<std::string> keyColumns = {
      "c9", "c7", "c1", "c10", "c3", "c8", "c0", "c6", "c2", "c4", "c5"};

  const auto expected = projectKeyColumns(input, keyColumns);
  for (const auto& sortOrder :
       {velox::core::kAscNullsFirst,
        velox::core::kAscNullsLast,
        velox::core::kDescNullsFirst,
        velox::core::kDescNullsLast}) {
    SCOPED_TRACE(
        fmt::format(
            "sortOrder={}{}",
            sortOrder.isAscending() ? "ASC" : "DESC",
            sortOrder.isNullsFirst() ? "_NULLS_FIRST" : "_NULLS_LAST"));
    const auto decoded = encodeAndDecode(
        input, keyColumns, repeatedSortOrders(keyColumns.size(), sortOrder));
    velox::test::assertEqualVectors(expected, decoded);
  }
}

TEST_F(KeyDecoderTest, roundTripWithMixedSortOrders) {
  auto input = makeRowVector(
      {makeNullableFlatVector<int64_t>({7, 3, std::nullopt}),
       makeNullableFlatVector<StringView>(
           {stringView("a"), stringView("b"), std::nullopt}, VARCHAR()),
       makeNullableFlatVector<Timestamp>(
           {Timestamp(1, 10), Timestamp(2, 20), std::nullopt}),
       makeNullableFlatVector<float>({1.5f, std::nullopt, -2.5f}),
       makeNullableFlatVector<int32_t>({10, std::nullopt, -10}, DATE())});

  const std::vector<std::string> keyColumns = {"c4", "c1", "c0", "c3", "c2"};
  const std::vector<velox::core::SortOrder> sortOrders = {
      velox::core::kDescNullsLast,
      velox::core::kAscNullsFirst,
      velox::core::kDescNullsFirst,
      velox::core::kAscNullsLast,
      velox::core::kDescNullsLast,
  };

  const auto decoded = encodeAndDecode(input, keyColumns, sortOrders);
  const auto expected = projectKeyColumns(input, keyColumns);
  velox::test::assertEqualVectors(expected, decoded);
}

TEST_F(KeyDecoderTest, fuzzRoundTripSupportedKeyTypes) {
  const auto rowType =
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"},
          {BIGINT(),
           INTEGER(),
           SMALLINT(),
           TINYINT(),
           DOUBLE(),
           REAL(),
           BOOLEAN(),
           VARCHAR(),
           VARBINARY(),
           TIMESTAMP(),
           DATE()});

  const std::vector<std::string> keyColumns = {
      "c8", "c0", "c10", "c7", "c3", "c9", "c1", "c6", "c2", "c4", "c5"};
  const std::vector<std::vector<velox::core::SortOrder>> sortOrderSets = {
      repeatedSortOrders(keyColumns.size(), velox::core::kAscNullsFirst),
      repeatedSortOrders(keyColumns.size(), velox::core::kAscNullsLast),
      repeatedSortOrders(keyColumns.size(), velox::core::kDescNullsFirst),
      repeatedSortOrders(keyColumns.size(), velox::core::kDescNullsLast),
      {velox::core::kDescNullsLast,
       velox::core::kAscNullsFirst,
       velox::core::kDescNullsFirst,
       velox::core::kAscNullsLast,
       velox::core::kDescNullsLast,
       velox::core::kAscNullsFirst,
       velox::core::kDescNullsFirst,
       velox::core::kAscNullsLast,
       velox::core::kDescNullsLast,
       velox::core::kAscNullsFirst,
       velox::core::kDescNullsFirst},
  };

  VectorFuzzer::Options options;
  options.vectorSize = 64;
  options.nullRatio = 0.2;
  options.stringLength = 32;
  options.stringVariableLength = true;
  options.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kNanoSeconds;
  VectorFuzzer fuzzer(options, pool_.get());

  constexpr int32_t kNumIterations = 20;
  for (int32_t iteration = 0; iteration < kNumIterations; ++iteration) {
    SCOPED_TRACE(fmt::format("iteration={}", iteration));
    const auto input = fuzzer.fuzzInputFlatRow(rowType);
    const auto expected = projectKeyColumns(input, keyColumns);

    for (size_t sortOrderIndex = 0; sortOrderIndex < sortOrderSets.size();
         ++sortOrderIndex) {
      SCOPED_TRACE(fmt::format("sortOrderIndex={}", sortOrderIndex));
      const auto& sortOrders = sortOrderSets.at(sortOrderIndex);
      const auto encodedKeys = encodeToStrings(input, keyColumns, sortOrders);
      auto decoder =
          KeyDecoder::create(keyColumns, rowType, sortOrders, pool_.get());
      auto& keyDecoder = checkedReference(decoder);

      const auto decoded =
          keyDecoder.decode(std::span<const std::string>(encodedKeys));
      velox::test::assertEqualVectors(expected, decoded);
      EXPECT_EQ(encodedKeys, encodeToStrings(decoded, keyColumns, sortOrders));
    }
  }
}

TEST_F(KeyDecoderTest, floatingPointValuesAreCanonicalized) {
  auto input = makeRowVector(
      {makeFlatVector<double>(
           {bitsToDouble(0x8000000000000000ULL),
            bitsToDouble(0x7ff0000000000001ULL),
            1.5}),
       makeFlatVector<float>(
           {bitsToFloat(0x80000000U), bitsToFloat(0x7f800001U), -2.5f})});

  const auto decoded = encodeAndDecode(
      input,
      {"c0", "c1"},
      {velox::core::kAscNullsFirst, velox::core::kAscNullsFirst});

  const auto* doubleValues = decoded->childAt(0)->asFlatVector<double>();
  const auto* floatValues = decoded->childAt(1)->asFlatVector<float>();

  EXPECT_EQ(doubleToBits(doubleValues->valueAt(0)), 0);
  EXPECT_EQ(floatToBits(floatValues->valueAt(0)), 0);

  EXPECT_EQ(doubleToBits(doubleValues->valueAt(1)), 0x7ff8000000000000ULL);
  EXPECT_EQ(floatToBits(floatValues->valueAt(1)), 0x7fc00000U);

  EXPECT_EQ(doubleValues->valueAt(2), 1.5);
  EXPECT_EQ(floatValues->valueAt(2), -2.5f);
}

TEST_F(KeyDecoderTest, malformedEncodedKeysThrow) {
  auto decoder = KeyDecoder::create(
      {"c0", "c1"},
      ROW({"c0", "c1"}, {BIGINT(), VARCHAR()}),
      {velox::core::kAscNullsFirst, velox::core::kAscNullsFirst},
      pool_.get());
  auto& keyDecoder = checkedReference(decoder);

  {
    const std::vector<std::string_view> encodedKeys = {
        std::string_view("\x01\x80", 2)};
    VELOX_ASSERT_THROW(
        keyDecoder.decode(std::span<const std::string_view>(encodedKeys)),
        "Malformed");
  }

  {
    const std::vector<std::string_view> encodedKeys = {
        std::string_view("\x01\x80\x00", 3)};
    VELOX_ASSERT_THROW(
        keyDecoder.decode(std::span<const std::string_view>(encodedKeys)),
        "Malformed");
  }

  {
    const std::vector<std::string_view> encodedKeys = {std::string_view(
        "\x01\x80\x00\x00\x00\x00\x00\x00\x00\x01\x01\x02\x00", 13)};
    VELOX_ASSERT_THROW(
        keyDecoder.decode(std::span<const std::string_view>(encodedKeys)),
        "invalid escaped string byte");
  }
}

} // namespace
} // namespace facebook::velox::serializer::test
