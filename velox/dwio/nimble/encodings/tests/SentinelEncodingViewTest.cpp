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
#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"

#include <array>
#include <bit>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"

namespace facebook::nimble::test {

class SentinelEncodingViewTest : public EncodingViewTest {
 protected:
  template <typename T>
  static typename TypeTraits<T>::physicalType physicalValue(T value) {
    if constexpr (std::is_floating_point_v<T>) {
      return std::bit_cast<typename TypeTraits<T>::physicalType>(value);
    } else {
      return value;
    }
  }

  // Builds the legacy Sentinel layout around a real Trivial child encoding.
  template <typename T>
  std::string encode(
      const std::vector<std::optional<T>>& rows,
      T sentinel,
      const Encoding::Options& options,
      CompressionType compression = CompressionType::Uncompressed) {
    Vector<T> values{pool_.get()};
    uint32_t numNulls{0};
    for (const auto& row : rows) {
      values.push_back(row.value_or(sentinel));
      numNulls += !row.has_value();
    }
    const auto child = Encoder<TrivialEncoding<T>>::encode(
        *buffer_, values, compression, options);
    using PhysicalType = typename TypeTraits<T>::physicalType;
    const auto sentinelSize = [&] {
      if constexpr (std::is_same_v<T, std::string_view>) {
        return sizeof(uint32_t) + sentinel.size();
      } else {
        return sizeof(PhysicalType);
      }
    }();
    const auto numRows = static_cast<uint32_t>(rows.size());
    std::string encoded(
        EncodingPrefix::serializedSize(numRows, options.useVarintRowCount) +
            2 * sizeof(uint32_t) + child.size() + sentinelSize,
        '\0');
    char* position = encoded.data();
    EncodingPrefix::serialize(
        EncodingType::Sentinel,
        TypeTraits<T>::dataType,
        numRows,
        options.useVarintRowCount,
        position);
    encoding::writeUint32(numNulls, position);
    encoding::writeUint32(static_cast<uint32_t>(child.size()), position);
    encoding::writeBytes(child, position);
    encoding::write<PhysicalType>(physicalValue(sentinel), position);
    EXPECT_EQ(position, encoded.data() + encoded.size());
    return encoded;
  }

  template <typename T>
  void verifyReads(const std::vector<std::optional<T>>& rows, T sentinel) {
    using PhysicalType = typename TypeTraits<T>::physicalType;
    const auto numRows = static_cast<uint32_t>(rows.size());
    for (const bool useVarint : {false, true}) {
      SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
      for (const auto compression :
           {CompressionType::Uncompressed, CompressionType::Zstd}) {
        SCOPED_TRACE(fmt::format("compression={}", compression));
        const Encoding::Options options{.useVarintRowCount = useVarint};
        const auto encoded = encode(rows, sentinel, options, compression);
        auto view = createEncodingView(encoded, pool_.get(), options);
        ASSERT_EQ(view->encodingType(), EncodingType::Sentinel);
        ASSERT_EQ(view->dataType(), TypeTraits<T>::dataType);
        ASSERT_EQ(view->rowCount(), numRows);
        auto* typedView = dynamic_cast<TypedEncodingView<T>*>(view.get());
        ASSERT_NE(typedView, nullptr);
        for (uint32_t row{0}; row < numRows; ++row) {
          SCOPED_TRACE(row);
          if (rows[row].has_value()) {
            PhysicalType actual{};
            view->readAt(row, &actual);
            EXPECT_EQ(actual, physicalValue(*rows[row]));
            EXPECT_EQ(
                physicalValue(typedView->readAt(row)),
                physicalValue(*rows[row]));
          } else {
            PhysicalType actual{};
            NIMBLE_ASSERT_THROW(
                view->readAt(row, &actual),
                "Sentinel EncodingView physical reads require a non-null row");
          }
        }

        const PhysicalType guard = [&]() -> PhysicalType {
          if constexpr (std::is_same_v<T, std::string_view>) {
            return "guard";
          } else {
            return static_cast<PhysicalType>(17);
          }
        }();
        const auto verifySelected = [&](const std::vector<uint32_t>& indices,
                                        auto read) {
          Vector<PhysicalType> output{pool_.get(), indices.size() + 2};
          std::fill(output.begin(), output.end(), guard);
          std::vector<uint32_t> actualNulls;
          const auto numNonNulls = read(
              [&](uint32_t index) { actualNulls.push_back(index); },
              output.data() + 1);
          std::vector<PhysicalType> expected;
          std::vector<uint32_t> expectedNulls;
          for (uint32_t i{0}; i < indices.size(); ++i) {
            const auto& row = rows[indices[i]];
            expected.push_back(
                row.has_value() ? physicalValue(*row) : PhysicalType{});
            if (!row.has_value()) {
              expectedNulls.push_back(i);
            }
          }
          EXPECT_EQ(numNonNulls, indices.size() - expectedNulls.size());
          EXPECT_EQ(actualNulls, expectedNulls);
          EXPECT_EQ(
              std::vector<PhysicalType>(output.begin() + 1, output.end() - 1),
              expected);
          EXPECT_EQ(output[0], guard);
          EXPECT_EQ(output[output.size() - 1], guard);
        };
        Vector<PhysicalType> physicalOutput{pool_.get(), numRows + 2};
        std::fill(physicalOutput.begin(), physicalOutput.end(), guard);
        view->read(0, numRows, physicalOutput.data() + 1);
        for (uint32_t row{0}; row < numRows; ++row) {
          SCOPED_TRACE(row);
          EXPECT_EQ(
              physicalOutput[row + 1],
              rows[row].has_value() ? physicalValue(*rows[row])
                                    : PhysicalType{});
        }
        EXPECT_EQ(physicalOutput[0], guard);
        EXPECT_EQ(physicalOutput[physicalOutput.size() - 1], guard);
        view->read(numRows, 0, physicalOutput.data() + numRows + 1);
        EXPECT_EQ(physicalOutput[physicalOutput.size() - 1], guard);
        for (const auto length : {0U, 1U, numRows - 1, numRows}) {
          SCOPED_TRACE(fmt::format("length={}", length));
          const std::array<RowRange, 1> range{{{0, length}}};
          std::vector<uint32_t> indices;
          for (uint32_t row{0}; row < length; ++row) {
            indices.push_back(row);
          }
          verifySelected(indices, [&](auto setNull, void* output) {
            return length == 0
                ? view->read(std::span<const RowRange>{}, setNull, output)
                : view->read(range, setNull, output);
          });
        }
        const std::vector<uint32_t> indices{numRows - 1, 0, 1, 1, 2};
        verifySelected(indices, [&](auto setNull, void* output) {
          return view->read(indices, setNull, output);
        });
        const std::array<RowRange, 2> ranges{
            RowRange{0, numRows / 2}, RowRange{numRows / 2 + 1, numRows}};
        std::vector<uint32_t> rangeIndices;
        for (const auto& range : ranges) {
          for (auto row = range.startRow; row < range.endRow; ++row) {
            rangeIndices.push_back(row);
          }
        }
        verifySelected(rangeIndices, [&](auto setNull, void* output) {
          return view->read(ranges, setNull, output);
        });
      }
    }
  }

  template <typename T>
  void verifyNumeric() {
    SCOPED_TRACE(fmt::format("dataType={}", TypeTraits<T>::dataType));
    T sentinel;
    if constexpr (std::is_floating_point_v<T>) {
      sentinel = std::bit_cast<T>(
          std::numeric_limits<typename TypeTraits<T>::physicalType>::max());
    } else {
      sentinel = std::numeric_limits<T>::max();
    }
    std::vector<std::optional<T>> rows;
    for (uint32_t row{0}; row < 131; ++row) {
      rows.emplace_back(
          row % 7 == 0 || row == 130
              ? std::nullopt
              : std::optional<T>{static_cast<T>(row % 51)});
    }
    verifyReads(rows, sentinel);
    verifyReads(std::vector<std::optional<T>>(131, T{3}), sentinel);
    verifyReads(std::vector<std::optional<T>>(131, std::nullopt), sentinel);
  }
};

TEST_F(SentinelEncodingViewTest, readsNumericTypes) {
  verifyNumeric<int8_t>();
  verifyNumeric<uint8_t>();
  verifyNumeric<int16_t>();
  verifyNumeric<uint16_t>();
  verifyNumeric<int32_t>();
  verifyNumeric<uint32_t>();
  verifyNumeric<int64_t>();
  verifyNumeric<uint64_t>();
  verifyNumeric<float>();
  verifyNumeric<double>();
}

TEST_F(SentinelEncodingViewTest, readsBooleanAndStringValues) {
  verifyReads<bool>({false, std::nullopt, false, false, std::nullopt}, true);
  verifyReads<std::string_view>(
      {"", "alpha", std::nullopt, "longer string value", std::nullopt},
      "<null>");
}

TEST_F(SentinelEncodingViewTest, readsZeroSentinel) {
  verifyReads<int64_t>({1, std::nullopt, 3, 4, std::nullopt}, 0);
}

TEST_F(SentinelEncodingViewTest, preservesFloatingPointBits) {
  std::vector<std::optional<float>> rows;
  const std::array<float, 4> values{
      0.0f, -0.0f, std::bit_cast<float>(uint32_t{0x7fc00001}), 2.5f};
  for (uint32_t row{0}; row < 131; ++row) {
    rows.emplace_back(
        row % 7 == 0 ? std::nullopt : std::optional<float>{values[row % 4]});
  }
  verifyReads(rows, std::bit_cast<float>(uint32_t{0xffffffff}));
}

TEST_F(SentinelEncodingViewTest, validatesPayloadBounds) {
  const std::vector<std::optional<int64_t>> rows{10, std::nullopt, 12};
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(useVarint);
    const Encoding::Options options{.useVarintRowCount = useVarint};
    const auto encoded = encode<int64_t>(rows, -1, options);
    const auto prefixSize = EncodingPrefix::serializedSize(3, useVarint);
    const auto truncated = encoded.substr(0, prefixSize + sizeof(uint32_t));
    NIMBLE_ASSERT_THROW(
        createEncodingView(truncated, pool_.get(), options),
        "Sentinel encoding is missing its null count or value-stream length");
    auto invalid = encoded;
    char* position = invalid.data() + prefixSize;
    encoding::writeUint32(4, position);
    NIMBLE_ASSERT_THROW(
        createEncodingView(invalid, pool_.get(), options),
        "Sentinel null count exceeds row count");
    invalid = encoded;
    position = invalid.data() + prefixSize + sizeof(uint32_t);
    encoding::writeUint32(std::numeric_limits<uint32_t>::max(), position);
    NIMBLE_ASSERT_THROW(
        createEncodingView(invalid, pool_.get(), options),
        "Sentinel value encoding exceeds payload bounds");
    invalid = encoded;
    position = invalid.data();
    EncodingPrefix::serialize(
        EncodingType::Sentinel, DataType::Int64, 4, useVarint, position);
    NIMBLE_ASSERT_THROW(
        createEncodingView(invalid, pool_.get(), options),
        "Sentinel child row count must match its parent");
  }
}

TEST_F(SentinelEncodingViewTest, validatesStringSentinelBounds) {
  const std::vector<std::optional<std::string_view>> rows{
      "alpha", std::nullopt, "beta"};
  auto encoded = encode<std::string_view>(rows, "<null>", {});
  char* position = encoded.data() + encoded.size() - sizeof(uint32_t) - 6;
  encoding::writeUint32(std::numeric_limits<uint32_t>::max(), position);
  NIMBLE_ASSERT_THROW(
      createEncodingView(encoded, pool_.get()),
      "Sentinel string exceeds payload bounds");
}

} // namespace facebook::nimble::test
