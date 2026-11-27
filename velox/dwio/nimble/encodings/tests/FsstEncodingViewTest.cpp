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
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"
#include "velox/dwio/nimble/encodings/views/DecodedStringBuffers.h"

namespace facebook::nimble::test {
namespace {

// Returns whether every non-empty string of 'values' lies in one of 'buffers'.
bool liesIn(
    std::span<const std::string_view> values,
    std::span<const velox::BufferPtr> buffers) {
  return std::all_of(values.begin(), values.end(), [&](std::string_view value) {
    return value.empty() ||
        std::any_of(
               buffers.begin(),
               buffers.end(),
               [&](const velox::BufferPtr& buffer) {
                 const auto* begin = buffer->as<char>();
                 return value.data() >= begin &&
                     value.data() + value.size() <= begin + buffer->size();
               });
  });
}

// Returns 'buffers' without repeats, in no particular order.
std::vector<velox::BufferPtr> distinct(std::vector<velox::BufferPtr> buffers) {
  std::sort(buffers.begin(), buffers.end());
  buffers.erase(std::unique(buffers.begin(), buffers.end()), buffers.end());
  return buffers;
}

// Returns an addStringBuffer for a read that appends each buffer the read
// passes to 'buffers'.
std::function<void(velox::BufferPtr)> appendTo(
    std::vector<velox::BufferPtr>& buffers) {
  return [&buffers](velox::BufferPtr buffer) {
    buffers.push_back(std::move(buffer));
  };
}

class FsstEncodingViewTest : public EncodingViewTest {
 protected:
  struct FsstSections {
    std::string_view prefix;
    CompressionType compressionType;
    std::string_view symbolTable;
    std::string_view lengths;
    std::string_view decodedLengths;
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
    cursor = remaining.data();
    const auto decodedLengthsSize = varint::readVarint32(&cursor);
    remaining.remove_prefix(cursor - remaining.data());
    const auto decodedLengths = remaining.substr(0, decodedLengthsSize);
    remaining.remove_prefix(decodedLengthsSize);
    return {
        .prefix = encoded.substr(0, prefixSize),
        .compressionType = compressionType,
        .symbolTable = symbolTable,
        .lengths = lengths,
        .decodedLengths = decodedLengths,
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
    appendSection(sections.decodedLengths);
    rebuilt.append(blob);
    return rebuilt;
  }

  // The layout written before decoded lengths existed: no decoded lengths
  // section and no flag in the compression type byte.
  std::string rebuildWithoutDecodedLengths(const FsstSections& sections) {
    std::string rebuilt{sections.prefix};
    rebuilt.push_back(
        static_cast<char>(
            static_cast<uint8_t>(sections.compressionType) &
            ~FsstEncoding::kDecodedLengthsFlag));
    for (const auto section : {sections.symbolTable, sections.lengths}) {
      char encodedSize[5];
      char* cursor = encodedSize;
      varint::writeVarint(static_cast<uint32_t>(section.size()), &cursor);
      rebuilt.append(encodedSize, cursor);
      rebuilt.append(section);
    }
    rebuilt.append(sections.blob);
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

  // Rejects at construction without caching, and in the first read with it.
  void expectMalformedView(
      std::string_view malformed,
      const char* expectedMessage) {
    NIMBLE_ASSERT_THROW(
        readFirstRow(createEncodingView(
            malformed,
            pool_.get(),
            {.cacheEncodingView = cacheEncodingView()})),
        expectedMessage);
  }

  static void readFirstRow(const std::unique_ptr<EncodingView>& view) {
    std::string_view value;
    view->readAt(0, &value);
  }

  // A cached view validates the whole blob in its first read. A view without
  // caching validates each string when a read reaches it.
  void expectMalformedBlob(
      std::string_view malformed,
      const char* expectedMessage) {
    if (cacheEncodingView()) {
      NIMBLE_ASSERT_THROW(
          readFirstRow(createEncodingView(
              malformed, pool_.get(), {.cacheEncodingView = true})),
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

TEST_P(FsstEncodingViewTest, readsEncodingsWithoutDecodedLengths) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 2'051);
  const std::vector<uint32_t> positions{
      0, 1, 2, 1'022, 1'023, 1'024, 1'025, 2'047, 2'048, 2'050};
  auto options = forceFsstOptions();
  options.cacheEncodingView = cacheEncodingView();

  const auto older =
      rebuildWithoutDecodedLengths(splitFsst(encodeFsst(values)));
  EXPECT_TRUE(FsstEncoding::decodedLengthsEncoding(older, {}).empty());
  expectEncodedReads(older, values, positions, options);
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
  EXPECT_EQ(
      view->read(
          ranges,
          [](uint32_t) {},
          /*addStringBuffer=*/nullptr,
          selected.data()),
      4);
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
  EXPECT_EQ(
      view->read(
          ranges,
          [](uint32_t) {},
          /*addStringBuffer=*/nullptr,
          output.data()),
      rows.size());
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

TEST_P(FsstEncodingViewTest, readDecodesItsChunksIntoOneBuffer) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 5'000);
  const auto encoded = encodeFsst(values);
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});

  // Reads the rows of 'range' with an addStringBuffer, and returns the
  // distinct buffers that the read passes, after checking that they hold
  // every string it returns. A cached view takes the rows of decoded chunks
  // from the chunks, and passes the buffers of the chunks.
  const auto passedBuffers = [&](RowRange range) {
    std::vector<std::string_view> strings(range.numRows());
    std::vector<velox::BufferPtr> buffers;
    view->read(
        std::array<RowRange, 1>{range},
        [](uint32_t) {},
        appendTo(buffers),
        strings.data());
    EXPECT_TRUE(liesIn(strings, buffers));
    return distinct(std::move(buffers));
  };

  // Asks for every row of chunks 0 and 1, and for a few rows of chunk 2, which
  // a cached view decodes alone, into the view itself.
  const std::array<RowRange, 2> ranges{
      RowRange{0, 2'048}, RowRange{2'100, 2'104}};
  std::vector<std::string_view> output(2'052);
  EXPECT_EQ(
      view->read(
          ranges,
          [](uint32_t) {},
          /*addStringBuffer=*/nullptr,
          output.data()),
      output.size());
  for (uint32_t row = 0; row < 2'048; ++row) {
    EXPECT_EQ(output[row], values[row]) << "row " << row;
  }
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(output[2'048 + i], values[2'100 + i]) << "row " << 2'100 + i;
  }

  // Reads every row of chunks 3 and 4 in one range, which a cached view
  // decodes into a buffer of their own.
  std::vector<std::string_view> tail(values.size() - 3'072);
  view->read(3'072, static_cast<uint32_t>(tail.size()), tail.data());
  for (uint32_t i = 0; i < tail.size(); ++i) {
    EXPECT_EQ(tail[i], values[3'072 + i]) << "row " << 3'072 + i;
  }
  if (!cacheEncodingView()) {
    return;
  }

  // One buffer holds chunks 0 and 1, and another one chunks 3 and 4.
  const auto headBuffers = passedBuffers(RowRange{0, 2'048});
  const auto tailBuffers =
      passedBuffers(RowRange{3'072, static_cast<uint32_t>(values.size())});
  ASSERT_EQ(headBuffers.size(), 1);
  ASSERT_EQ(tailBuffers.size(), 1);
  EXPECT_NE(headBuffers[0], tailBuffers[0]);
}

TEST_P(FsstEncodingViewTest, viewsDecodeIntoTheBuffersTheyShare) {
  // Few enough rows that the strings of both views fit in the first buffer,
  // of 32 KB.
  std::vector<std::string> storage;
  const auto values = makeValues(storage, 400);

  // Two chunks of one stream, each with a view of its own.
  Vector<std::string_view> firstValues{pool_.get()};
  Vector<std::string_view> secondValues{pool_.get()};
  for (uint32_t row = 0; row < values.size(); ++row) {
    (row < 250 ? firstValues : secondValues).push_back(values[row]);
  }
  const auto firstEncoded = encodeFsst(firstValues);
  const auto secondEncoded = encodeFsst(secondValues);

  DecodedStringBuffers decodedStringBuffers{pool_.get()};
  Encoding::Options options{.cacheEncodingView = cacheEncodingView()};
  options.decodedStringBuffers = &decodedStringBuffers;
  const auto firstView = createEncodingView(firstEncoded, pool_.get(), options);
  const auto secondView =
      createEncodingView(secondEncoded, pool_.get(), options);

  // Reads every row of each view, as a scan of the stream does.
  std::vector<std::string_view> firstOutput(firstValues.size());
  firstView->read(0, firstView->rowCount(), firstOutput.data());
  std::vector<std::string_view> secondOutput(secondValues.size());
  secondView->read(0, secondView->rowCount(), secondOutput.data());
  EXPECT_TRUE(
      std::equal(firstOutput.begin(), firstOutput.end(), firstValues.begin()));
  EXPECT_TRUE(
      std::equal(
          secondOutput.begin(), secondOutput.end(), secondValues.begin()));

  // A view without caching decodes each row alone, into the view itself.
  if (!cacheEncodingView()) {
    return;
  }

  // The strings of both views lie in one buffer: reads of every row of each
  // view given an addStringBuffer take the strings from the decoded chunks
  // and pass that buffer.
  std::vector<velox::BufferPtr> buffers;
  for (const auto* view : {firstView.get(), secondView.get()}) {
    std::vector<std::string_view> strings(view->rowCount());
    view->read(
        std::array<RowRange, 1>{RowRange{0, view->rowCount()}},
        [](uint32_t) {},
        appendTo(buffers),
        strings.data());
  }
  buffers = distinct(std::move(buffers));
  ASSERT_EQ(buffers.size(), 1);
  EXPECT_TRUE(liesIn(firstOutput, buffers));
  EXPECT_TRUE(liesIn(secondOutput, buffers));
}

TEST_P(
    FsstEncodingViewTest,
    readWithAddStringBufferDecodesRowsIntoBuffersOfItsOwn) {
  std::vector<std::string> storage;
  const auto values = makeValues(storage, 5'000);
  const auto encoded = encodeFsst(values);
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});

  // Asks for every row of chunks 0 and 1, for part of chunk 2, and for few
  // enough rows of chunk 4 that a read without addStringBuffer decodes them
  // alone.
  const std::array<RowRange, 3> ranges{
      RowRange{0, 2'048}, RowRange{2'100, 2'600}, RowRange{4'990, 5'000}};
  std::vector<std::string_view> output(2'048 + 500 + 10);
  std::vector<velox::BufferPtr> addedBuffers;
  EXPECT_EQ(
      view->read(
          ranges, [](uint32_t) {}, appendTo(addedBuffers), output.data()),
      output.size());
  uint32_t outputOffset{0};
  for (const auto& range : ranges) {
    for (auto row = range.startRow; row < range.endRow; ++row) {
      EXPECT_EQ(output[outputOffset++], values[row]) << "row " << row;
    }
  }

  // The buffers that the read passes hold every string of the read.
  ASSERT_FALSE(addedBuffers.empty());
  EXPECT_TRUE(liesIn(output, addedBuffers));

  // The read cached nothing, so the same read decodes the rows again, into
  // buffers of its own.
  std::vector<std::string_view> again(output.size());
  std::vector<velox::BufferPtr> againBuffers;
  view->read(ranges, [](uint32_t) {}, appendTo(againBuffers), again.data());
  EXPECT_EQ(again, output);
  EXPECT_TRUE(liesIn(again, againBuffers));
  for (const auto& buffer : againBuffers) {
    EXPECT_EQ(
        std::find(addedBuffers.begin(), addedBuffers.end(), buffer),
        addedBuffers.end());
  }
}

TEST_P(
    FsstEncodingViewTest,
    readOfRowsWithAddStringBufferDecodesIntoOneBuffer) {
  std::vector<std::string> storage;
  const auto values = makeValues(storage, 4'096);
  const auto encoded = encodeFsst(values);
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});

  // Asks for a few rows of each chunk, from the last to the first, too few for
  // a cached view to decode the chunks, which a read without addStringBuffer
  // decodes alone, into the view itself.
  std::vector<uint32_t> rows;
  rows.reserve(42);
  for (uint32_t i = 0; i < 42; ++i) {
    rows.push_back(4'095 - i * 97);
  }
  std::vector<std::string_view> output(rows.size());
  std::vector<velox::BufferPtr> addedBuffers;
  EXPECT_EQ(
      view->read(
          rows, [](uint32_t) {}, appendTo(addedBuffers), output.data()),
      rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(output[i], values[rows[i]]) << "row " << rows[i];
  }

  // One buffer holds every string of the read.
  ASSERT_EQ(addedBuffers.size(), 1);
  EXPECT_TRUE(liesIn(output, addedBuffers));
}

TEST_P(FsstEncodingViewTest, readWithAddStringBufferSkipsStringsDecodedAlone) {
  std::vector<std::string> storage;
  const auto values = makeValues(storage, 2'048);
  const auto encoded = encodeFsst(values);
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});

  // A read without addStringBuffer of a few rows decodes them alone, into the
  // view itself, where no buffer holds them.
  const std::array<uint32_t, 3> rows{6, 900, 1'504};
  std::array<std::string_view, 3> alone{};
  view->readAt(rows, alone.data());
  EXPECT_THAT(
      alone, ::testing::ElementsAre(values[6], values[900], values[1'504]));

  // A read of the same rows given an addStringBuffer decodes them again, into
  // a buffer that it passes, rather than take them from the view.
  std::array<std::string_view, 3> output{};
  std::vector<velox::BufferPtr> addedBuffers;
  EXPECT_EQ(
      view->read(
          rows, [](uint32_t) {}, appendTo(addedBuffers), output.data()),
      rows.size());
  EXPECT_THAT(
      output, ::testing::ElementsAre(values[6], values[900], values[1'504]));
  ASSERT_EQ(addedBuffers.size(), 1);
  EXPECT_TRUE(liesIn(output, addedBuffers));
  EXPECT_FALSE(liesIn(alone, addedBuffers));
}

TEST_P(
    FsstEncodingViewTest,
    readWithAddStringBufferTakesStringsOfDecodedChunks) {
  std::vector<std::string> storage;
  const auto values = makeValues(storage, 3'072);
  const auto encoded = encodeFsst(values);
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});

  // A read without addStringBuffer of every row of chunk 1, which a cached
  // view decodes and caches.
  std::vector<std::string_view> chunk(1'024);
  view->read(1'024, 1'024, chunk.data());

  // A read that spans the decoded chunk takes the strings of its rows from it,
  // decodes chunk 0, of which it asks for every row, and decodes the few rows
  // of chunk 2 that it asks for into a buffer of its own, instead of into the
  // view itself. It passes the buffers of the chunks and of its rows.
  const std::array<RowRange, 2> ranges{
      RowRange{0, 2'048}, RowRange{2'100, 2'104}};
  std::vector<std::string_view> output(2'052);
  std::vector<velox::BufferPtr> addedBuffers;
  view->read(ranges, [](uint32_t) {}, appendTo(addedBuffers), output.data());
  EXPECT_TRUE(
      std::equal(output.begin(), output.begin() + 2'048, values.begin()));
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(output[2'048 + i], values[2'100 + i]) << "row " << 2'100 + i;
  }
  EXPECT_TRUE(liesIn(output, addedBuffers));

  // A view without caching decodes every row of the read into its region,
  // whose buffers it passes.
  if (!cacheEncodingView()) {
    return;
  }

  // The region, then the buffers of chunks 0 and 1, which two reads decoded.
  EXPECT_EQ(addedBuffers.size(), 3);
  for (uint32_t i = 0; i < chunk.size(); ++i) {
    EXPECT_EQ(output[1'024 + i].data(), chunk[i].data()) << "row " << 1'024 + i;
  }

  // The rows of chunk 2 lie in the region, in neither chunk buffer: a read
  // of chunks 0 and 1 alone takes their rows from the decoded chunks and
  // passes only the chunk buffers.
  std::vector<std::string_view> chunks(2'048);
  std::vector<velox::BufferPtr> chunkBuffers;
  view->read(
      std::array<RowRange, 1>{RowRange{0, 2'048}},
      [](uint32_t) {},
      appendTo(chunkBuffers),
      chunks.data());
  EXPECT_TRUE(liesIn(chunks, chunkBuffers));
  EXPECT_TRUE(liesIn(std::span{output}.first(2'048), chunkBuffers));
  EXPECT_FALSE(liesIn(std::span{output}.subspan(2'048), chunkBuffers));
}

TEST_P(FsstEncodingViewTest, readWithAddStringBufferSpansBuffers) {
  // FSST encodes each run of eight repeated bytes as one code, so these
  // strings decode to about eight times their compressed size, about 600 KB.
  std::vector<std::string> storage;
  storage.reserve(2'048);
  Vector<std::string_view> values{pool_.get()};
  for (uint32_t row = 0; row < 2'048; ++row) {
    storage.emplace_back(
        std::string(256 + row % 64, 'a') + std::to_string(row % 10));
    values.push_back(storage.back());
  }
  const auto encoded = encodeFsst(values);

  // Reads every row, so that the region of the read fills the buffer it starts
  // in and goes on in new ones, whose strings stay where they are decoded.
  const auto expectReadKeepsStrings = [&](const Encoding::Options& options) {
    const auto view = createEncodingView(encoded, pool_.get(), options);
    const std::array<RowRange, 1> ranges{RowRange{0, view->rowCount()}};
    std::vector<std::string_view> output(values.size());
    std::vector<velox::BufferPtr> addedBuffers;
    view->read(ranges, [](uint32_t) {}, appendTo(addedBuffers), output.data());
    EXPECT_TRUE(std::equal(output.begin(), output.end(), values.begin()));
    EXPECT_GT(addedBuffers.size(), 1);
    EXPECT_TRUE(liesIn(output, addedBuffers));
  };
  {
    SCOPED_TRACE("buffer of the view's own");
    expectReadKeepsStrings({.cacheEncodingView = cacheEncodingView()});
  }
  {
    SCOPED_TRACE("shared buffers");
    DecodedStringBuffers decodedStringBuffers{pool_.get()};
    Encoding::Options options{.cacheEncodingView = cacheEncodingView()};
    options.decodedStringBuffers = &decodedStringBuffers;
    expectReadKeepsStrings(options);
  }
}

TEST_P(FsstEncodingViewTest, readsWithAddStringBufferFillTheBuffersViewsShare) {
  // Few enough rows that the strings of both views fit in the first buffer,
  // of 32 KB.
  std::vector<std::string> storage;
  const auto values = makeValues(storage, 400);
  Vector<std::string_view> firstValues{pool_.get()};
  Vector<std::string_view> secondValues{pool_.get()};
  for (uint32_t row = 0; row < values.size(); ++row) {
    (row < 250 ? firstValues : secondValues).push_back(values[row]);
  }
  const auto firstEncoded = encodeFsst(firstValues);
  const auto secondEncoded = encodeFsst(secondValues);

  DecodedStringBuffers decodedStringBuffers{pool_.get()};
  Encoding::Options options{.cacheEncodingView = cacheEncodingView()};
  options.decodedStringBuffers = &decodedStringBuffers;
  const auto firstView = createEncodingView(firstEncoded, pool_.get(), options);
  const auto secondView =
      createEncodingView(secondEncoded, pool_.get(), options);

  // Reads every row of each view, as a scan of the stream does.
  const auto readAll = [](const EncodingView& view,
                          std::vector<std::string_view>& output,
                          std::vector<velox::BufferPtr>& addedBuffers) {
    const std::array<RowRange, 1> ranges{RowRange{0, view.rowCount()}};
    output.resize(view.rowCount());
    view.read(ranges, [](uint32_t) {}, appendTo(addedBuffers), output.data());
  };
  std::vector<std::string_view> firstOutput;
  std::vector<velox::BufferPtr> firstBuffers;
  readAll(*firstView, firstOutput, firstBuffers);
  std::vector<std::string_view> secondOutput;
  std::vector<velox::BufferPtr> secondBuffers;
  readAll(*secondView, secondOutput, secondBuffers);
  EXPECT_TRUE(
      std::equal(firstOutput.begin(), firstOutput.end(), firstValues.begin()));
  EXPECT_TRUE(
      std::equal(
          secondOutput.begin(), secondOutput.end(), secondValues.begin()));
  EXPECT_TRUE(liesIn(firstOutput, firstBuffers));
  EXPECT_TRUE(liesIn(secondOutput, secondBuffers));

  // The strings of both reads lie in one buffer.
  ASSERT_EQ(firstBuffers.size(), 1);
  EXPECT_EQ(secondBuffers, firstBuffers);
  EXPECT_TRUE(liesIn(firstOutput, firstBuffers));
  EXPECT_TRUE(liesIn(secondOutput, firstBuffers));
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

TEST_P(FsstEncodingViewTest, supportsConcurrentReadsOfEveryRow) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 5'000);
  const auto encoded = encodeFsst(values);
  const auto view = createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});
  const auto numRows = static_cast<uint32_t>(values.size());

  // Every thread reads every row at once, through a range or a list of
  // ranges, so threads that decode the same chunks race to cache them.
  expectConcurrent([&](int threadIndex) {
    std::vector<std::string_view> output(numRows);
    if (threadIndex % 2 == 0) {
      view->read(0, numRows, output.data());
    } else {
      const std::array<RowRange, 1> ranges{RowRange{0, numRows}};
      view->read(
          ranges,
          [](uint32_t) {},
          /*addStringBuffer=*/nullptr,
          output.data());
    }
    return std::equal(output.begin(), output.end(), values.begin());
  });
}

TEST_P(FsstEncodingViewTest, supportsConcurrentReadsWithAddStringBuffer) {
  std::vector<std::string> storage;
  const auto values = makeValues(storage, 5'000);
  const auto encoded = encodeFsst(values);

  // The views decode into buffers of their own: views that share
  // DecodedStringBuffers must not be read at once.
  const Encoding::Options options{.cacheEncodingView = cacheEncodingView()};
  std::vector<std::unique_ptr<EncodingView>> views;
  views.reserve(4);
  for (int i = 0; i < 4; ++i) {
    views.emplace_back(createEncodingView(encoded, pool_.get(), options));
  }
  const auto numRows = static_cast<uint32_t>(values.size());

  // Two threads read each view at once. One decodes every row for its own
  // read, while the other decodes and caches the chunks, which the first then
  // takes the strings of its rows from if they are cached before it starts.
  expectConcurrent([&](int threadIndex) {
    const auto& view = views[threadIndex / 2 % views.size()];
    std::vector<std::string_view> output(numRows);
    if (threadIndex % 2 == 1) {
      view->read(0, numRows, output.data());
      return std::equal(output.begin(), output.end(), values.begin());
    }

    // A read given an addStringBuffer passes buffers that hold all of its
    // strings.
    const std::array<RowRange, 1> ranges{RowRange{0, numRows}};
    std::vector<velox::BufferPtr> addedBuffers;
    view->read(ranges, [](uint32_t) {}, appendTo(addedBuffers), output.data());
    return std::equal(output.begin(), output.end(), values.begin()) &&
        !addedBuffers.empty() && liesIn(output, addedBuffers);
  });
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

TEST_P(FsstEncodingViewTest, readsLongStrings) {
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

  // Reads every row, which decodes the chunks that hold the long strings.
  std::vector<std::string_view> all(values.size());
  view->read(0, view->rowCount(), all.data());
  for (uint32_t row = 0; row < values.size(); ++row) {
    EXPECT_EQ(all[row], values[row]) << "row " << row;
  }
}

TEST_P(FsstEncodingViewTest, readsStringsThatExpandBeyondTheChunkEstimate) {
  // FSST encodes each run of eight repeated bytes as one code, so these
  // strings decode to about eight times their compressed size, more than a
  // cached view of an encoding without decoded lengths first allocates for a
  // chunk.
  std::vector<std::string> storage;
  storage.reserve(2'048);
  Vector<std::string_view> values{pool_.get()};
  for (uint32_t row = 0; row < 2'048; ++row) {
    storage.emplace_back(
        std::string(256 + row % 64, 'a') + std::to_string(row % 10));
    values.push_back(storage.back());
  }
  const auto encoded = encodeFsst(values);
  const auto older = rebuildWithoutDecodedLengths(splitFsst(encoded));
  for (const auto& layout : {encoded, older}) {
    const auto view = createEncodingView(
        layout, pool_.get(), {.cacheEncodingView = cacheEncodingView()});
    std::vector<std::string_view> output(values.size());
    view->read(0, view->rowCount(), output.data());
    for (uint32_t row = 0; row < values.size(); ++row) {
      EXPECT_EQ(output[row], values[row]) << "row " << row;
    }
  }
}

TEST_P(FsstEncodingViewTest, readsChunksOfEmptyStrings) {
  std::vector<std::string> storage;
  auto values = makeValues(storage, 3'000);
  for (uint32_t row = 1'024; row < 2'048; ++row) {
    storage[row].clear();
    values[row] = storage[row];
  }
  const auto encoded = encodeFsst(values);
  const auto older = rebuildWithoutDecodedLengths(splitFsst(encoded));
  for (const auto& layout : {encoded, older}) {
    const auto view = createEncodingView(
        layout, pool_.get(), {.cacheEncodingView = cacheEncodingView()});
    std::vector<std::string_view> output(values.size());
    view->read(0, view->rowCount(), output.data());
    for (uint32_t row = 0; row < values.size(); ++row) {
      EXPECT_EQ(output[row], values[row]) << "row " << row;
    }
  }
}

TEST_P(FsstEncodingViewTest, readsRangesGivenAnAddStringBuffer) {
  std::vector<std::string> storage;
  const auto values = makeValues(storage, 5'000);
  const auto encoded = encodeFsst(values);
  const auto older = rebuildWithoutDecodedLengths(splitFsst(encoded));

  // Reads 'ranges' with an addStringBuffer, and checks the strings and that the
  // buffers that the read passes hold them.
  const auto expectRanges = [&](const EncodingView& view,
                                std::span<const RowRange> ranges) {
    uint32_t numRows{0};
    for (const auto& range : ranges) {
      numRows += range.numRows();
    }
    std::vector<std::string_view> output(numRows);
    std::vector<velox::BufferPtr> buffers;
    EXPECT_EQ(
        view.read(
            ranges, [](uint32_t) {}, appendTo(buffers), output.data()),
        numRows);
    uint32_t index{0};
    for (const auto& range : ranges) {
      for (auto row = range.startRow; row < range.endRow; ++row, ++index) {
        EXPECT_EQ(output[index], values[row]) << "row " << row;
      }
    }
    EXPECT_TRUE(liesIn(output, buffers));
  };

  for (const auto& layout : {encoded, older}) {
    const auto view = createEncodingView(
        layout, pool_.get(), {.cacheEncodingView = cacheEncodingView()});

    // A few rows, which a cached view decodes one string at a time.
    expectRanges(*view, std::array<RowRange, 1>{RowRange{7, 11}});

    // Every row, and ranges that start and end inside chunks, and cross chunk
    // boundaries.
    expectRanges(*view, std::array<RowRange, 1>{RowRange{0, 5'000}});
    expectRanges(
        *view,
        std::array<RowRange, 4>{
            RowRange{10, 1'030},
            RowRange{1'100, 1'104},
            RowRange{2'000, 4'100},
            RowRange{4'990, 5'000}});

    // Chunk 1 is decoded, so a read takes its rows from the chunk and decodes
    // the rows of the other chunks.
    std::vector<std::string_view> chunk(1'024);
    view->read(1'024, 1'024, chunk.data());
    expectRanges(*view, std::array<RowRange, 1>{RowRange{0, 5'000}});
    expectRanges(
        *view,
        std::array<RowRange, 2>{RowRange{500, 1'500}, RowRange{3'000, 3'600}});
  }
}

TEST_P(FsstEncodingViewTest, readsFewRowsWithoutDecodedLengths) {
  if (!cacheEncodingView()) {
    GTEST_SKIP() << "A view without caching does not read decoded lengths.";
  }
  std::vector<std::string> storage;
  auto values = makeValues(storage, 256);
  const auto encoded = encodeFsst(values);
  const auto sections = splitFsst(encoded);
  auto decodedLengths = decodeLengths(sections.decodedLengths);
  ASSERT_GT(values[0].size(), 0);

  // A decoded length that fits its compressed string, but not what the string
  // decodes to.
  ++decodedLengths[0];
  auto replaced = sections;
  const auto encodedDecodedLengths =
      encodeTrivialChild<uint32_t>(decodedLengths);
  replaced.decodedLengths = encodedDecodedLengths;
  const auto malformed = rebuildFsst(
      replaced, sections.symbolTable, sections.lengths, sections.blob);
  const auto view =
      createEncodingView(malformed, pool_.get(), {.cacheEncodingView = true});

  // Keeps the buffers that hold the strings that reads return.
  std::vector<velox::BufferPtr> buffers;
  const auto read = [&](RowRange range) {
    std::vector<std::string_view> output(range.numRows());
    view->read(
        std::array<RowRange, 1>{range},
        [](uint32_t) {},
        appendTo(buffers),
        output.data());
    return output;
  };

  // A read of a few rows decodes them one string at a time, and does not read
  // the decoded lengths, so it returns the strings.
  const auto few = read(RowRange{0, 4});
  for (uint32_t row = 0; row < few.size(); ++row) {
    EXPECT_EQ(few[row], values[row]) << "row " << row;
  }

  // A read of more rows decodes them with the decoded lengths, which do not
  // match the strings.
  for (int attempt = 0; attempt < 2; ++attempt) {
    NIMBLE_ASSERT_FILE_THROW(
        read(RowRange{0, 128}),
        "FSST decoded size does not match the decoded lengths.");
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

TEST_P(FsstEncodingViewTest, rejectsMalformedDecodedLengths) {
  if (!cacheEncodingView()) {
    GTEST_SKIP() << "A view without caching does not read decoded lengths.";
  }
  std::vector<std::string> storage;
  auto values = makeValues(storage, 256);
  const auto encoded = encodeFsst(values);
  const auto sections = splitFsst(encoded);
  const auto lengths = decodeLengths(sections.lengths);
  const auto decodedLengths = decodeLengths(sections.decodedLengths);
  ASSERT_EQ(decodedLengths.size(), values.size());
  ASSERT_GT(lengths[0], 0);

  // A view reads the decoded lengths when it first decodes a chunk, so the
  // view is created and the read of every row fails.
  const auto expectMalformed = [&](std::string_view encodedDecodedLengths,
                                   const char* expectedMessage) {
    auto replaced = sections;
    replaced.decodedLengths = encodedDecodedLengths;
    const auto malformed = rebuildFsst(
        replaced, sections.symbolTable, sections.lengths, sections.blob);
    const auto view =
        createEncodingView(malformed, pool_.get(), {.cacheEncodingView = true});
    std::vector<std::string_view> output(values.size());

    // A read that fails fails again.
    for (int attempt = 0; attempt < 2; ++attempt) {
      NIMBLE_ASSERT_FILE_THROW(
          view->read(0, view->rowCount(), output.data()), expectedMessage);
    }
  };

  std::vector<uint64_t> uint64DecodedLengths(
      decodedLengths.begin(), decodedLengths.end());
  expectMalformed(
      encodeTrivialChild<uint64_t>(uint64DecodedLengths),
      "FSST decoded lengths encoding must contain Uint32 values.");

  expectMalformed(
      encodeTrivialChild<uint32_t>(
          std::span<const uint32_t>{decodedLengths}.first(
              decodedLengths.size() - 1)),
      "FSST decoded lengths row count does not match the parent encoding.");

  expectMalformed(
      encodeNullableLengths(decodedLengths),
      "FSST decoded lengths encoding must not be nullable: Nullable");

  std::vector<std::string> text;
  text.reserve(decodedLengths.size());
  for (const auto length : decodedLengths) {
    text.push_back(std::to_string(length));
  }
  expectMalformed(
      encodeTrivialChild<std::string_view>(
          std::vector<std::string_view>(text.begin(), text.end())),
      "FSST decoded lengths encoding must contain Uint32 values.");

  {
    SCOPED_TRACE("a non-empty compressed string decodes to an empty one");
    auto replaced = decodedLengths;
    replaced[0] = 0;
    expectMalformed(
        encodeTrivialChild<uint32_t>(replaced),
        "FSST compressed length is too long for its decoded length.");
  }
  {
    SCOPED_TRACE("a code decodes to more than eight bytes");
    auto replaced = decodedLengths;
    replaced[0] = lengths[0] * 8 + 1;
    expectMalformed(
        encodeTrivialChild<uint32_t>(replaced),
        "FSST decoded length is too long for its compressed length.");
  }

  // A length that fits its compressed string but not what the string decodes
  // to.
  for (const uint32_t wrong : {decodedLengths[0] - 1, decodedLengths[0] + 1}) {
    auto replaced = decodedLengths;
    replaced[0] = wrong;
    expectMalformed(
        encodeTrivialChild<uint32_t>(replaced),
        "FSST decoded size does not match the decoded lengths.");
  }
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
        readFirstRow(createEncodingView(
            trailingBlobEncoding, pool_.get(), {.cacheEncodingView = true})),
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
