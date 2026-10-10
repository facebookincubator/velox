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

#include <algorithm>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"

namespace facebook::nimble::test {
namespace {

class FsstEncodingViewTest : public EncodingViewTest {
 protected:
  struct FsstSections {
    std::string_view prefix;
    CompressionType compressionType;
    std::string_view symbolTable;
    std::string_view lengths;
    std::string_view blob;
  };

  std::string encodeFsst(const Vector<std::string_view>& values) {
    return std::string{Encoder<FsstEncoding>::encode(
        *buffer_, values, CompressionType::Uncompressed, forceFsstOptions())};
  }

  FsstSections splitFsst(std::string_view encoded) {
    const auto prefixSize = EncodingPrefix::prefixSize(encoded, false);
    auto remaining = encoded.substr(prefixSize);
    const char* cursor = remaining.data();
    const auto compressionType =
        static_cast<CompressionType>(encoding::readChar(cursor));
    remaining.remove_prefix(cursor - remaining.data());
    cursor = remaining.data();
    const auto symbolTableSize = varint::readVarint32(&cursor);
    remaining.remove_prefix(cursor - remaining.data());
    const auto symbolTable = remaining.substr(0, symbolTableSize);
    remaining.remove_prefix(symbolTableSize);
    cursor = remaining.data();
    const auto lengthsSize = varint::readVarint32(&cursor);
    remaining.remove_prefix(cursor - remaining.data());
    const auto lengths = remaining.substr(0, lengthsSize);
    remaining.remove_prefix(lengthsSize);
    return {
        .prefix = encoded.substr(0, prefixSize),
        .compressionType = compressionType,
        .symbolTable = symbolTable,
        .lengths = lengths,
        .blob = remaining,
    };
  }

  std::string rebuildFsst(
      const FsstSections& sections,
      std::string_view symbolTable,
      std::string_view lengths,
      std::string_view blob) {
    std::string rebuilt{sections.prefix};
    const auto appendSection = [&](std::string_view section) {
      NIMBLE_CHECK_LE(section.size(), std::numeric_limits<uint32_t>::max());
      char encodedSize[5];
      char* cursor = encodedSize;
      varint::writeVarint(static_cast<uint32_t>(section.size()), &cursor);
      rebuilt.append(encodedSize, cursor);
      rebuilt.append(section);
    };
    rebuilt.push_back(static_cast<char>(sections.compressionType));
    appendSection(symbolTable);
    appendSection(lengths);
    rebuilt.append(blob);
    return rebuilt;
  }

  template <typename T>
  std::string encodeTrivialChild(std::span<const T> values) {
    Vector<T> input{pool_.get(), values.size()};
    std::copy(values.begin(), values.end(), input.begin());
    Buffer buffer{*pool_};
    return std::string{Encoder<TrivialEncoding<T>>::encode(buffer, input)};
  }

  std::string encodeNullableLengths(std::span<const uint32_t> values) {
    Vector<uint32_t> input{pool_.get(), values.size()};
    std::copy(values.begin(), values.end(), input.begin());
    Vector<bool> nonNulls{pool_.get(), values.size()};
    std::fill(nonNulls.begin(), nonNulls.end(), true);
    Buffer buffer{*pool_};
    return std::string{Encoder<NullableEncoding<uint32_t>>::encodeNullable(
        buffer, input, nonNulls)};
  }

  std::vector<uint32_t> decodeLengths(std::string_view encodedLengths) {
    auto encoding = EncodingFactory().create(
        *pool_, encodedLengths, [](uint32_t) -> void* { return nullptr; });
    NIMBLE_CHECK_NOT_NULL(encoding);
    std::vector<uint32_t> lengths(encoding->rowCount());
    encoding->materialize(
        static_cast<uint32_t>(lengths.size()), lengths.data());
    return lengths;
  }

  std::string buildIncompleteEscapeEncoding(
      const FsstSections& sections,
      size_t row) {
    const auto lengths = decodeLengths(sections.lengths);
    NIMBLE_CHECK_LT(row, lengths.size());
    NIMBLE_CHECK_GT(lengths[row], 0);
    size_t rowEnd{0};
    for (size_t i = 0; i <= row; ++i) {
      rowEnd += lengths[i];
    }

    std::string blob{sections.blob};
    if (lengths[row] >= 2 &&
        static_cast<uint8_t>(blob[rowEnd - 2]) == FSST_ESC) {
      blob[rowEnd - 2] = 0;
    }
    blob[rowEnd - 1] = static_cast<char>(FSST_ESC);
    return rebuildFsst(sections, sections.symbolTable, sections.lengths, blob);
  }

  void expectMalformedView(
      std::string_view malformed,
      const char* expectedMessage) {
    NIMBLE_ASSERT_THROW(
        createEncodingView(malformed, pool_.get()), expectedMessage);
  }

  Encoding::Options forceFsstOptions() const {
    return {.fsstCompressionTargetRatio = std::numeric_limits<double>::max()};
  }

  Vector<std::string_view> makeValues(
      std::vector<std::string>& storage,
      uint32_t rowCount) {
    storage.reserve(rowCount);
    for (uint32_t row = 0; row < rowCount; ++row) {
      switch (row % 5) {
        case 0:
          storage.emplace_back(
              "https://www.example.com/experiments/targeting/" +
              std::to_string(row % 37));
          break;
        case 1:
          storage.emplace_back(
              "qrt_exposure_info_v2:campaign:placement:country:" +
              std::to_string(row % 19));
          break;
        case 2:
          storage.emplace_back();
          break;
        case 3:
          storage.emplace_back("binary\0payload\0", 15);
          break;
        default:
          storage.emplace_back(
              "repeated production-shaped log message for FSST decoding");
          break;
      }
    }

    Vector<std::string_view> values{pool_.get()};
    values.reserve(storage.size());
    for (const auto& value : storage) {
      values.emplace_back(value);
    }
    return values;
  }
};

TEST_F(FsstEncodingViewTest, readsAcrossLazyChunkBoundaries) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 2'051);
  const std::vector<uint32_t> positions{
      0, 1, 2, 1'022, 1'023, 1'024, 1'025, 2'047, 2'048, 2'050};

  expectReads<FsstEncoding>(values, positions, forceFsstOptions());
  expectReads<FsstEncoding>(
      values, positions, forceFsstOptions(), CompressionType::Lz4);
}

TEST_F(FsstEncodingViewTest, supportsConcurrentReadsAcrossChunks) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 2'048);
  std::mt19937 rng{42};
  std::uniform_int_distribution<uint32_t> row{
      0, static_cast<uint32_t>(values.size() - 1)};
  std::vector<uint32_t> positions;
  positions.reserve(4'096);
  for (uint32_t i = 0; i < 4'096; ++i) {
    positions.push_back(row(rng));
  }

  expectConcurrentReads<FsstEncoding>(values, positions, forceFsstOptions());
}

TEST_F(FsstEncodingViewTest, rejectsMalformedMetadata) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 256);
  const auto encoded = encodeFsst(values);
  ASSERT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Fsst);
  const auto sections = splitFsst(encoded);
  ASSERT_GE(sections.symbolTable.size(), 17);
  const auto lengths = decodeLengths(sections.lengths);
  ASSERT_EQ(lengths.size(), values.size());

  const auto truncatedSymbolTable = rebuildFsst(
      sections,
      sections.symbolTable.substr(0, 16),
      sections.lengths,
      sections.blob);
  expectMalformedView(truncatedSymbolTable, "Truncated FSST symbol table.");

  std::vector<uint64_t> uint64Lengths(lengths.begin(), lengths.end());
  const auto wrongTypeLengths = encodeTrivialChild<uint64_t>(uint64Lengths);
  const auto wrongTypeEncoding = rebuildFsst(
      sections, sections.symbolTable, wrongTypeLengths, sections.blob);
  expectMalformedView(
      wrongTypeEncoding, "FSST lengths encoding must contain Uint32 values.");

  const auto shortLengths = encodeTrivialChild<uint32_t>(
      std::span<const uint32_t>{lengths}.first(lengths.size() - 1));
  const auto wrongRowCountEncoding =
      rebuildFsst(sections, sections.symbolTable, shortLengths, sections.blob);
  expectMalformedView(
      wrongRowCountEncoding,
      "FSST lengths row count does not match the parent encoding.");

  const auto nullableLengths = encodeNullableLengths(lengths);
  const auto nullableEncoding = rebuildFsst(
      sections, sections.symbolTable, nullableLengths, sections.blob);
  expectMalformedView(
      nullableEncoding, "FSST lengths encoding must not be nullable.");
}

TEST_F(FsstEncodingViewTest, rejectsMalformedBlob) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 256);
  const auto encoded = encodeFsst(values);
  ASSERT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Fsst);
  const auto sections = splitFsst(encoded);

  auto lengths = decodeLengths(sections.lengths);
  ASSERT_FALSE(lengths.empty());
  ASSERT_GT(lengths.back(), 0);
  ++lengths.back();
  const auto oversizedLengths = encodeTrivialChild<uint32_t>(lengths);
  const auto oversizedEncoding = rebuildFsst(
      sections, sections.symbolTable, oversizedLengths, sections.blob);
  expectMalformedView(
      oversizedEncoding, "FSST compressed length exceeds the remaining blob.");

  std::string blobWithTrailingByte{sections.blob};
  blobWithTrailingByte.push_back('\0');
  const auto trailingBlobEncoding = rebuildFsst(
      sections, sections.symbolTable, sections.lengths, blobWithTrailingByte);
  expectMalformedView(
      trailingBlobEncoding,
      "FSST compressed lengths do not match the blob size.");
}

TEST_F(FsstEncodingViewTest, rejectsIncompleteEscapeInBlob) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 256);
  const auto encoded = encodeFsst(values);
  ASSERT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Fsst);
  const auto malformed = buildIncompleteEscapeEncoding(splitFsst(encoded), 0);
  expectMalformedView(
      malformed, "FSST compressed string ends with an incomplete escape code.");
}

} // namespace
} // namespace facebook::nimble::test
