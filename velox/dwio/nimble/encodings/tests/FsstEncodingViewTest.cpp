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
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
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

  // Rejects at construction, with or without caching.
  void expectMalformedView(
      std::string_view malformed,
      const char* expectedMessage) {
    NIMBLE_ASSERT_THROW(
        createEncodingView(
            malformed, pool_.get(), {.cacheEncodingView = cacheEncodingView()}),
        expectedMessage);
  }

  // A cached view validates the whole blob when the view is created. A view
  // without caching validates each string when a read reaches it.
  void expectMalformedBlob(
      std::string_view malformed,
      const char* expectedMessage) {
    if (cacheEncodingView()) {
      NIMBLE_ASSERT_THROW(
          createEncodingView(
              malformed, pool_.get(), {.cacheEncodingView = true}),
          expectedMessage);
      return;
    }
    auto view = createEncodingView(
        malformed, pool_.get(), {.cacheEncodingView = false});
    std::vector<std::string_view> output(view->rowCount());
    NIMBLE_ASSERT_THROW(
        view->read(0, view->rowCount(), output.data()), expectedMessage);
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

TEST_P(FsstEncodingViewTest, readsAcrossLazyChunkBoundaries) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 2'051);
  const std::vector<uint32_t> positions{
      0, 1, 2, 1'022, 1'023, 1'024, 1'025, 2'047, 2'048, 2'050};

  expectReads<FsstEncoding>(values, positions, forceFsstOptions());
  expectReads<FsstEncoding>(
      values, positions, forceFsstOptions(), CompressionType::Lz4);
}

TEST_P(FsstEncodingViewTest, decodedChunksUseMemoryPool) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 256);
  const auto encoded = encodeFsst(values);
  velox::memory::MemoryManager manager;
  const auto rootPool = manager.addRootPool();
  const auto decodePool = rootPool->addLeafChild("fsst-decoded-chunks");
  ASSERT_TRUE(decodePool->trackUsage());
  const auto bytesBeforeView = decodePool->usedBytes();
  auto view = createEncodingView(
      encoded, decodePool.get(), {.cacheEncodingView = cacheEncodingView()});
  const auto bytesBeforeDecode = decodePool->usedBytes();
  size_t decodedBytes{0};
  for (const auto value : values) {
    decodedBytes += value.size();
  }

  // A cached view decodes the whole chunk, and a view without caching keeps
  // the strings that it decodes alone in blocks from the pool. Both settings
  // must release everything they took from the pool.
  std::vector<std::string_view> chunk(values.size());
  view->read(0, static_cast<uint32_t>(values.size()), chunk.data());
  for (uint32_t row = 0; row < values.size(); ++row) {
    EXPECT_EQ(chunk[row], values[row]) << "row " << row;
  }
  if (cacheEncodingView()) {
    EXPECT_GE(decodePool->usedBytes(), bytesBeforeDecode + decodedBytes);
  } else {
    EXPECT_GT(decodePool->usedBytes(), bytesBeforeDecode);
  }
  const auto bytesAfterDecode = decodePool->usedBytes();
  std::string_view actual;
  view->readAt(1, &actual);
  EXPECT_EQ(actual, values[1]);
  if (cacheEncodingView()) {
    EXPECT_EQ(decodePool->usedBytes(), bytesAfterDecode);
  }
  view.reset();
  EXPECT_EQ(decodePool->usedBytes(), bytesBeforeView);
}

TEST_P(FsstEncodingViewTest, sparseReadsDecodeOnlyTheirRows) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 4'096);
  const auto encoded = encodeFsst(values);
  velox::memory::MemoryManager manager;
  const auto rootPool = manager.addRootPool();
  const auto decodePool = rootPool->addLeafChild("fsst-sparse-reads");
  const auto view = createEncodingView(
      encoded, decodePool.get(), {.cacheEncodingView = cacheEncodingView()});
  const auto bytesBeforeReads = decodePool->usedBytes();

  // A take of a few rows of each chunk, through any read, decodes just those
  // rows, so even a cached view allocates no decoded chunk from the pool.
  size_t chunkBytes{0};
  for (uint32_t row = 0; row < 1'024; ++row) {
    chunkBytes += values[row].size();
  }
  std::string_view value;
  view->readAt(5, &value);
  EXPECT_EQ(value, values[5]);

  const std::vector<uint32_t> indices{3'070, 1'030, 2'050, 1'030};
  std::vector<std::string_view> indexed(indices.size());
  view->readAt(indices, indexed.data());
  for (size_t i = 0; i < indices.size(); ++i) {
    EXPECT_EQ(indexed[i], values[indices[i]]) << "index " << indices[i];
  }

  constexpr uint32_t kRangeRows = 4;
  std::vector<std::string_view> range(kRangeRows);
  view->read(2'100, kRangeRows, range.data());
  for (uint32_t i = 0; i < kRangeRows; ++i) {
    EXPECT_EQ(range[i], values[2'100 + i]) << "row " << 2'100 + i;
  }

  const std::array<RowRange, 2> ranges{
      RowRange{10, 13}, RowRange{4'000, 4'001}};
  std::vector<std::string_view> selected(4);
  EXPECT_EQ(view->read(ranges, [](uint32_t) {}, selected.data()), 4);
  EXPECT_EQ(selected[0], values[10]);
  EXPECT_EQ(selected[1], values[11]);
  EXPECT_EQ(selected[2], values[12]);
  EXPECT_EQ(selected[3], values[4'000]);

  EXPECT_LT(decodePool->usedBytes() - bytesBeforeReads, chunkBytes);
}

TEST_P(FsstEncodingViewTest, rowReadsDecodeChunkOnceTheyAskForEnoughOfIt) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 1'024);
  const auto encoded = encodeFsst(values);
  velox::memory::MemoryManager manager;
  const auto rootPool = manager.addRootPool();
  const auto decodePool = rootPool->addLeafChild("fsst-row-reads");
  const auto view = createEncodingView(
      encoded, decodePool.get(), {.cacheEncodingView = cacheEncodingView()});
  size_t decodedBytes{0};
  for (const auto value : values) {
    decodedBytes += value.size();
  }

  // Reads of one row at a time decode their rows alone, until they have asked
  // for enough rows of the chunk for a cached view to decode the whole chunk,
  // which allocates the decoded bytes of every row of the chunk in one read.
  std::vector<std::string_view> rows(values.size());
  std::optional<uint32_t> firstRowAfterChunkDecode;
  for (uint32_t row = 0; row < values.size(); ++row) {
    const auto bytesBeforeRead = decodePool->usedBytes();
    view->readAt(row, &rows[row]);
    if (!firstRowAfterChunkDecode.has_value() &&
        decodePool->usedBytes() >= bytesBeforeRead + decodedBytes) {
      firstRowAfterChunkDecode = row;
    }
  }
  if (cacheEncodingView()) {
    ASSERT_TRUE(firstRowAfterChunkDecode.has_value());
    EXPECT_GT(*firstRowAfterChunkDecode, 0);
    EXPECT_LT(*firstRowAfterChunkDecode, values.size() / 2);
  } else {
    EXPECT_FALSE(firstRowAfterChunkDecode.has_value());
  }

  // The strings of the rows read before the chunk was decoded stay valid.
  for (uint32_t row = 0; row < values.size(); ++row) {
    EXPECT_EQ(rows[row], values[row]) << "row " << row;
  }
}

TEST_P(FsstEncodingViewTest, rangeReadDecidesOnceForAllItsRowsOfChunk) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 3'500);
  const auto encoded = encodeFsst(values);
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});

  // Asks for a few rows of chunks 0 and 3, most rows of chunk 1, and every
  // other row of chunk 2 in ranges of one row.
  std::vector<RowRange> ranges{{5, 6}, {1'000, 2'000}};
  for (uint32_t row = 2'048; row < 3'072; row += 2) {
    ranges.emplace_back(row, row + 1);
  }
  ranges.emplace_back(3'100, 3'101);
  ranges.emplace_back(3'400, 3'401);
  std::vector<uint32_t> rows;
  for (const auto& range : ranges) {
    for (auto row = range.startRow; row < range.endRow; ++row) {
      rows.push_back(row);
    }
  }
  std::vector<std::string_view> output(rows.size());
  EXPECT_EQ(view->read(ranges, [](uint32_t) {}, output.data()), rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(output[i], values[rows[i]]) << "row " << rows[i];
  }

  // The read took each row from where later reads of the row take it: from
  // the decoded chunk for chunks 1 and 2, of which it asked for enough rows,
  // and from the string decoded for the row alone for chunks 0 and 3.
  for (size_t i = 0; i < rows.size(); ++i) {
    if (output[i].empty()) {
      continue;
    }
    std::string_view again;
    view->readAt(rows[i], &again);
    EXPECT_EQ(again.data(), output[i].data()) << "row " << rows[i];
  }
}

TEST_P(FsstEncodingViewTest, supportsConcurrentReadsAcrossChunks) {
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

TEST_P(FsstEncodingViewTest, returnedStringsStayValidAcrossReads) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 256);
  const auto encoded = encodeFsst(values);
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});
  std::string_view first;
  view->readAt(17, &first);

  // A row read again returns the same bytes, so the view decoded it once.
  std::string_view again;
  view->readAt(17, &again);
  EXPECT_EQ(first.data(), again.data());
  // Callers such as MainlyConstant fill one output from several reads of a
  // view, so a string must outlive the reads that follow it, including a read
  // of its own row and the decode of its chunk.
  for (uint32_t row = 0; row < values.size(); ++row) {
    std::string_view value;
    view->readAt(row, &value);
    EXPECT_EQ(value, values[row]);
  }
  EXPECT_EQ(first, values[17]);
}

TEST_P(FsstEncodingViewTest, readsScatteredAndRepeatedIndices) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 256);
  const auto encoded = encodeFsst(values);
  const std::vector<uint32_t> indices{255, 3, 130, 3, 0, 200, 255};
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});
  std::vector<std::string_view> output(indices.size());
  view->readAt(indices, output.data());
  for (size_t i = 0; i < indices.size(); ++i) {
    EXPECT_EQ(output[i], values[indices[i]]) << "index " << indices[i];
  }
}

TEST_P(FsstEncodingViewTest, readsLongStringsAlone) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 2'048);
  // Each of these rows decodes to more bytes than all the short rows of the
  // view together.
  std::string longValue;
  for (uint32_t i = 0; i < 100'000; ++i) {
    longValue.push_back(static_cast<char>((i * 7) % 256));
  }
  for (const auto row : {7, 8, 1'500}) {
    values[row] = longValue;
  }
  const auto encoded = encodeFsst(values);
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});

  // Asks for few enough rows of each chunk that a cached view decodes them
  // alone too.
  const std::vector<uint32_t> indices{6, 7, 8, 9, 1'499, 1'500, 1'501};
  std::vector<std::string_view> output(indices.size());
  view->readAt(indices, output.data());
  std::string_view value;
  view->readAt(2'000, &value);
  EXPECT_EQ(value, values[2'000]);
  for (size_t i = 0; i < indices.size(); ++i) {
    EXPECT_EQ(output[i], values[indices[i]]) << "index " << indices[i];
  }
}

TEST_P(FsstEncodingViewTest, readsStringsThatExpandBeyondTheChunkEstimate) {
  // FSST encodes each run of eight repeated bytes as one code, so these
  // strings decode to about eight times their compressed size, more than a
  // cached view first allocates for a chunk.
  std::vector<std::string> storage;
  storage.reserve(2'048);
  Vector<std::string_view> values{pool_.get()};
  for (uint32_t row = 0; row < 2'048; ++row) {
    storage.emplace_back(
        std::string(256 + row % 64, 'a') + std::to_string(row % 10));
    values.push_back(storage.back());
  }
  const auto encoded = encodeFsst(values);
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});
  std::vector<std::string_view> output(values.size());
  view->read(0, view->rowCount(), output.data());
  for (uint32_t row = 0; row < values.size(); ++row) {
    EXPECT_EQ(output[row], values[row]) << "row " << row;
  }
}

TEST_P(FsstEncodingViewTest, rejectsOutOfRangeIndices) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 256);
  const auto encoded = encodeFsst(values);
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});
  const std::vector<uint32_t> indices{3, 256};
  std::vector<std::string_view> output(indices.size());
  NIMBLE_ASSERT_THROW(
      view->readAt(indices, output.data()), "Read index out of range");
}

TEST_P(FsstEncodingViewTest, readsZeroTerminatedSymbolTable) {
  std::vector<std::string> storage{
      "", "alpha", "repeated production-shaped value"};
  for (auto& value : storage) {
    value.push_back('\0');
  }

  std::vector<size_t> inputLengths;
  std::vector<const unsigned char*> inputPointers;
  inputLengths.reserve(storage.size());
  inputPointers.reserve(storage.size());
  for (const auto& value : storage) {
    inputLengths.push_back(value.size());
    inputPointers.push_back(
        reinterpret_cast<const unsigned char*>(value.data()));
  }

  std::unique_ptr<fsst_encoder_t, decltype(&nimble_fsst_destroy)> encoder{
      nimble_fsst_create(
          storage.size(), inputLengths.data(), inputPointers.data(), 1),
      &nimble_fsst_destroy};
  ASSERT_NE(encoder, nullptr);

  std::array<unsigned char, FSST_MAXHEADER> symbolTable{};
  const auto symbolTableSize =
      nimble_fsst_export(encoder.get(), symbolTable.data());
  ASSERT_GT(symbolTableSize, 0);

  const auto totalInputBytes =
      std::accumulate(inputLengths.begin(), inputLengths.end(), size_t{0});
  std::vector<unsigned char> compressedBytes(
      totalInputBytes * 2 + storage.size() * 7);
  std::vector<size_t> compressedLengths(storage.size());
  std::vector<unsigned char*> compressedPointers(storage.size());
  ASSERT_EQ(
      nimble_fsst_compress(
          encoder.get(),
          storage.size(),
          inputLengths.data(),
          inputPointers.data(),
          compressedBytes.size(),
          compressedBytes.data(),
          compressedLengths.data(),
          compressedPointers.data()),
      storage.size());

  std::string blob;
  std::vector<uint32_t> encodedLengths;
  encodedLengths.reserve(storage.size());
  for (size_t row = 0; row < storage.size(); ++row) {
    ASSERT_LE(compressedLengths[row], std::numeric_limits<uint32_t>::max());
    encodedLengths.push_back(static_cast<uint32_t>(compressedLengths[row]));
    blob.append(
        reinterpret_cast<const char*>(compressedPointers[row]),
        compressedLengths[row]);
  }

  Vector<std::string_view> values{pool_.get()};
  values.reserve(storage.size());
  for (const auto& value : storage) {
    values.emplace_back(value);
  }
  const auto templateEncoding = encodeFsst(values);
  const auto sections = splitFsst(templateEncoding);
  const auto lengthsEncoding = encodeTrivialChild<uint32_t>(encodedLengths);
  const auto encoded = rebuildFsst(
      sections,
      std::string_view{
          reinterpret_cast<const char*>(symbolTable.data()), symbolTableSize},
      lengthsEncoding,
      blob);

  auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});
  ASSERT_NE(view, nullptr);
  for (uint32_t row = 0; row < values.size(); ++row) {
    std::string_view actual;
    view->readAt(row, &actual);
    EXPECT_EQ(actual, values[row]);
  }
}

TEST_P(FsstEncodingViewTest, rejectsMalformedMetadata) {
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
      nullableEncoding, "FSST lengths encoding must not be nullable: Nullable");
}

TEST_P(FsstEncodingViewTest, rejectsMalformedBlob) {
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
  expectMalformedBlob(
      oversizedEncoding, "FSST compressed length exceeds the remaining blob.");

  std::string blobWithTrailingByte{sections.blob};
  blobWithTrailingByte.push_back('\0');
  const auto trailingBlobEncoding = rebuildFsst(
      sections, sections.symbolTable, sections.lengths, blobWithTrailingByte);
  // Only a cached view checks every length against the blob size.
  if (cacheEncodingView()) {
    NIMBLE_ASSERT_THROW(
        createEncodingView(
            trailingBlobEncoding, pool_.get(), {.cacheEncodingView = true}),
        "FSST compressed lengths do not match the blob size.");
  }
}

TEST_P(FsstEncodingViewTest, rejectsIncompleteEscapeInBlob) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 256);
  const auto encoded = encodeFsst(values);
  ASSERT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Fsst);
  const auto malformed = buildIncompleteEscapeEncoding(splitFsst(encoded), 0);
  expectMalformedBlob(
      malformed, "FSST compressed string ends with an incomplete escape code.");
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(FsstEncodingViewTest);

} // namespace
} // namespace facebook::nimble::test
