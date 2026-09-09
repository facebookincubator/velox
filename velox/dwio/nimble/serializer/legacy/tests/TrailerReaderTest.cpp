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

#include <algorithm>
#include <bit>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/serializer/Options.h"
#include "velox/dwio/nimble/serializer/legacy/TrailerReader.h"

namespace facebook::nimble::serde::legacy {
namespace {

// Builds a legacy Trivial-encoded kLegacyCompact trailer for the given sizes:
//   [encodingByte=Trivial][raw u32 array][trailerSize:u32]
std::string buildLegacyTrivialTrailer(const std::vector<uint32_t>& sizes) {
  std::string buffer;
  buffer.push_back(static_cast<char>(EncodingType::Trivial));
  buffer.append(
      reinterpret_cast<const char*>(sizes.data()),
      sizes.size() * sizeof(uint32_t));
  const uint32_t trailerSize = static_cast<uint32_t>(buffer.size());
  buffer.append(reinterpret_cast<const char*>(&trailerSize), sizeof(uint32_t));
  return buffer;
}

// Writes a varint32 onto the back of |buf| (matches master's writeVarint).
void appendVarint32(std::string& buf, uint32_t value) {
  while (value >= 128) {
    buf.push_back(static_cast<char>((value & 127) | 128));
    value >>= 7;
  }
  buf.push_back(static_cast<char>(value));
}

// Bit-packs |values| at |bitWidth| in the same little-endian layout that
// FixedBitArray::bulkGet32 expects (matches master's bulkSet32 byte order).
// Buffer size matches FixedBitArray::bufferSize.
std::string packFixedBitWidth(
    const std::vector<uint32_t>& values,
    int bitWidth) {
  if (values.empty()) {
    // bitWidth==0 OR empty input: emit only the slop bytes.
    return std::string(7, '\0');
  }
  const uint64_t totalBits = values.size() * static_cast<uint64_t>(bitWidth);
  const uint64_t bufSize = ((totalBits + 7) / 8) + /*slop=*/7;
  std::string buf(bufSize, '\0');
  if (bitWidth == 0) {
    return buf;
  }
  uint64_t bitsOffset = 0;
  for (const auto v : values) {
    const uint64_t byteOffset = bitsOffset >> 3;
    const uint64_t bitsRemainder = bitsOffset & 7;
    uint64_t word;
    std::memcpy(&word, buf.data() + byteOffset, sizeof(word));
    word |= (static_cast<uint64_t>(v) << bitsRemainder);
    std::memcpy(buf.data() + byteOffset, &word, sizeof(word));
    bitsOffset += static_cast<uint64_t>(bitWidth);
  }
  return buf;
}

// Builds a legacy MainlyConstant-encoded kLegacyCompact trailer for the given
// (streamCount, indices, sizes) shape. Mirrors master's writer wire format:
//   [encodingByte=MainlyConstant]
//   [streamCount:varint][nonZeroCount:varint]
//   (only when nonZeroCount > 0:)
//     [idxBitWidth:1B][packed indices]
//     [valBitWidth:1B][packed values]
//   [trailerSize:u32]
//
// The early-return on `nonZeroCount == 0` is part of the master decoder's
// invariant; the writer must not emit bit-width bytes in that case or the
// post-decode `NIMBLE_CHECK_EQ(payload, trailerEnd)` will mismatch.
std::string buildLegacyMainlyConstantTrailer(
    uint32_t streamCount,
    const std::vector<uint32_t>& indices,
    const std::vector<uint32_t>& sizes) {
  EXPECT_EQ(indices.size(), sizes.size());
  std::string buffer;
  buffer.push_back(static_cast<char>(EncodingType::MainlyConstant));
  appendVarint32(buffer, streamCount);
  appendVarint32(buffer, static_cast<uint32_t>(indices.size()));

  if (!indices.empty()) {
    const auto idxBitWidth = static_cast<uint8_t>(
        std::bit_width(*std::max_element(indices.begin(), indices.end())));
    buffer.push_back(static_cast<char>(idxBitWidth));
    buffer.append(packFixedBitWidth(indices, idxBitWidth));

    const auto valBitWidth = static_cast<uint8_t>(
        std::bit_width(*std::max_element(sizes.begin(), sizes.end())));
    buffer.push_back(static_cast<char>(valBitWidth));
    buffer.append(packFixedBitWidth(sizes, valBitWidth));
  }

  const uint32_t trailerSize = static_cast<uint32_t>(buffer.size());
  buffer.append(reinterpret_cast<const char*>(&trailerSize), sizeof(uint32_t));
  return buffer;
}

TEST(LegacyTrailerReaderTest, readLegacyTrailerStreamMetadataTrivialSparse) {
  // Dense input has zeros that should be filtered out by the
  // normalize-to-sparse step inside the legacy reader.
  const std::vector<uint32_t> denseSizes = {0, 17, 0, 0, 42, 99, 0};
  const auto buffer = buildLegacyTrivialTrailer(denseSizes);

  const auto* end = buffer.data() + buffer.size();
  auto [streamIndices, streamSizes] = readLegacyTrailerStreamMetadata(end);

  ASSERT_EQ(streamIndices.size(), streamSizes.size());
  ASSERT_EQ(streamIndices.size(), 3u);
  EXPECT_EQ(streamIndices[0], 1u);
  EXPECT_EQ(streamIndices[1], 4u);
  EXPECT_EQ(streamIndices[2], 5u);
  EXPECT_EQ(streamSizes[0], 17u);
  EXPECT_EQ(streamSizes[1], 42u);
  EXPECT_EQ(streamSizes[2], 99u);
}

TEST(LegacyTrailerReaderTest, readLegacyTrailerStreamMetadataTrivialDense) {
  // All-non-zero input: every slot appears in the normalized sparse pair.
  const std::vector<uint32_t> denseSizes = {1, 2, 3, 4, 5};
  const auto buffer = buildLegacyTrivialTrailer(denseSizes);

  const auto* end = buffer.data() + buffer.size();
  auto [streamIndices, streamSizes] = readLegacyTrailerStreamMetadata(end);

  ASSERT_EQ(streamIndices.size(), 5u);
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(streamIndices[i], i);
    EXPECT_EQ(streamSizes[i], denseSizes[i]);
  }
}

TEST(LegacyTrailerReaderTest, readLegacyTrailerStreamMetadataTrivialEmpty) {
  // Empty input -> empty trailer + empty sparse output.
  const std::vector<uint32_t> denseSizes;
  const auto buffer = buildLegacyTrivialTrailer(denseSizes);

  const auto* end = buffer.data() + buffer.size();
  auto [streamIndices, streamSizes] = readLegacyTrailerStreamMetadata(end);

  EXPECT_TRUE(streamIndices.empty());
  EXPECT_TRUE(streamSizes.empty());
}

TEST(
    LegacyTrailerReaderTest,
    readLegacyTrailerStreamMetadataMainlyConstantSparse) {
  // Typical MainlyConstant case: 8 stream slots, 3 of them non-zero.
  const uint32_t streamCount = 8;
  const std::vector<uint32_t> indices = {1, 4, 5};
  const std::vector<uint32_t> sizes = {17, 42, 99};
  const auto buffer =
      buildLegacyMainlyConstantTrailer(streamCount, indices, sizes);

  const auto* end = buffer.data() + buffer.size();
  auto [streamIndices, streamSizes] = readLegacyTrailerStreamMetadata(end);

  ASSERT_EQ(streamIndices.size(), 3u);
  ASSERT_EQ(streamSizes.size(), 3u);
  EXPECT_EQ(streamIndices, indices);
  EXPECT_EQ(streamSizes, sizes);
}

TEST(
    LegacyTrailerReaderTest,
    readLegacyTrailerStreamMetadataMainlyConstantEmpty) {
  // nonZeroCount==0 path: the reader must short-circuit before reading any
  // bit-width bytes. streamCount stays at 0 here too — the most common shape
  // for an all-empty stripe in production.
  const auto buffer = buildLegacyMainlyConstantTrailer(
      /*streamCount=*/0, /*indices=*/{}, /*sizes=*/{});

  const auto* end = buffer.data() + buffer.size();
  auto [streamIndices, streamSizes] = readLegacyTrailerStreamMetadata(end);

  EXPECT_TRUE(streamIndices.empty());
  EXPECT_TRUE(streamSizes.empty());
}

TEST(
    LegacyTrailerReaderTest,
    readLegacyTrailerStreamMetadataMainlyConstantDense) {
  // Edge case: every slot non-zero. nonZeroCount == streamCount; the
  // NIMBLE_CHECK_LE(nonZeroCount, streamCount) bound must accept equality.
  const uint32_t streamCount = 4;
  const std::vector<uint32_t> indices = {0, 1, 2, 3};
  const std::vector<uint32_t> sizes = {7, 8, 9, 10};
  const auto buffer =
      buildLegacyMainlyConstantTrailer(streamCount, indices, sizes);

  const auto* end = buffer.data() + buffer.size();
  auto [streamIndices, streamSizes] = readLegacyTrailerStreamMetadata(end);

  EXPECT_EQ(streamIndices, indices);
  EXPECT_EQ(streamSizes, sizes);
}

TEST(
    LegacyTrailerReaderTest,
    readLegacyTrailerStreamMetadataMainlyConstantMaxBitWidth) {
  // Stress the bit-packer at the FixedBitWidth ceiling: sizes hold a
  // ~32-bit value so valBitWidth == 32 (the largest accepted by the
  // decoder's NIMBLE_CHECK_LE valBitWidth, 32 bound).
  const uint32_t streamCount = 1u << 30;
  const std::vector<uint32_t> indices = {0, (1u << 29), (1u << 30) - 1};
  const std::vector<uint32_t> sizes = {1u, 0xDEADBEEFu, 0xFFFFFFFFu / 2u};
  const auto buffer =
      buildLegacyMainlyConstantTrailer(streamCount, indices, sizes);

  const auto* end = buffer.data() + buffer.size();
  auto [streamIndices, streamSizes] = readLegacyTrailerStreamMetadata(end);

  EXPECT_EQ(streamIndices, indices);
  EXPECT_EQ(streamSizes, sizes);
}

// Builds a legacy Varint-encoded kLegacyCompact trailer:
//   [encodingByte=Varint][count:varint][v_0:varint]...[v_N:varint]
//   [trailerSize:u32]
std::string buildLegacyVarintTrailer(const std::vector<uint32_t>& denseSizes) {
  std::string buffer;
  buffer.push_back(static_cast<char>(EncodingType::Varint));
  appendVarint32(buffer, static_cast<uint32_t>(denseSizes.size()));
  for (const auto v : denseSizes) {
    appendVarint32(buffer, v);
  }
  const uint32_t trailerSize = static_cast<uint32_t>(buffer.size());
  buffer.append(reinterpret_cast<const char*>(&trailerSize), sizeof(uint32_t));
  return buffer;
}

// Builds a legacy Delta-encoded kLegacyCompact trailer:
//   [encodingByte=Delta][count:varint][first:varint][delta_1:varint]...
//   [trailerSize:u32]
std::string buildLegacyDeltaTrailer(const std::vector<uint32_t>& denseSizes) {
  std::string buffer;
  buffer.push_back(static_cast<char>(EncodingType::Delta));
  appendVarint32(buffer, static_cast<uint32_t>(denseSizes.size()));
  if (!denseSizes.empty()) {
    appendVarint32(buffer, denseSizes[0]);
    for (size_t i = 1; i < denseSizes.size(); ++i) {
      appendVarint32(buffer, denseSizes[i] - denseSizes[i - 1]);
    }
  }
  const uint32_t trailerSize = static_cast<uint32_t>(buffer.size());
  buffer.append(reinterpret_cast<const char*>(&trailerSize), sizeof(uint32_t));
  return buffer;
}

// Builds a legacy FixedBitWidth-encoded kLegacyCompact trailer:
//   [encodingByte=FixedBitWidth][bitWidth:1B][count:varint][packed bytes]
//   [trailerSize:u32]
std::string buildLegacyFixedBitWidthTrailer(
    const std::vector<uint32_t>& denseSizes) {
  std::string buffer;
  buffer.push_back(static_cast<char>(EncodingType::FixedBitWidth));
  const uint8_t bitWidth = denseSizes.empty()
      ? uint8_t{0}
      : static_cast<uint8_t>(std::bit_width(
            *std::max_element(denseSizes.begin(), denseSizes.end())));
  buffer.push_back(static_cast<char>(bitWidth));
  appendVarint32(buffer, static_cast<uint32_t>(denseSizes.size()));
  if (!denseSizes.empty()) {
    buffer.append(packFixedBitWidth(denseSizes, bitWidth));
  }
  const uint32_t trailerSize = static_cast<uint32_t>(buffer.size());
  buffer.append(reinterpret_cast<const char*>(&trailerSize), sizeof(uint32_t));
  return buffer;
}

TEST(LegacyTrailerReaderTest, readLegacyTrailerStreamMetadataVarint) {
  // Mix small + large varint-encoded sizes; zeros must scatter to sparse.
  const std::vector<uint32_t> denseSizes = {0, 5, 128, 0, 16384, 1u << 20, 0};
  const auto buffer = buildLegacyVarintTrailer(denseSizes);

  const auto* end = buffer.data() + buffer.size();
  auto [streamIndices, streamSizes] = readLegacyTrailerStreamMetadata(end);

  ASSERT_EQ(streamIndices.size(), 4u);
  EXPECT_EQ(streamIndices, (std::vector<uint32_t>{1, 2, 4, 5}));
  EXPECT_EQ(streamSizes, (std::vector<uint32_t>{5, 128, 16384, 1u << 20}));
}

TEST(LegacyTrailerReaderTest, readLegacyTrailerStreamMetadataDelta) {
  // Monotonically increasing dense sizes — Delta's natural happy path. No
  // zeros so every slot survives the scatter-to-sparse step.
  const std::vector<uint32_t> denseSizes = {10, 12, 15, 20, 100, 250};
  const auto buffer = buildLegacyDeltaTrailer(denseSizes);

  const auto* end = buffer.data() + buffer.size();
  auto [streamIndices, streamSizes] = readLegacyTrailerStreamMetadata(end);

  ASSERT_EQ(streamSizes.size(), 6u);
  EXPECT_EQ(streamSizes, denseSizes);
  EXPECT_EQ(streamIndices, (std::vector<uint32_t>{0, 1, 2, 3, 4, 5}));
}

TEST(LegacyTrailerReaderTest, readLegacyTrailerStreamMetadataFixedBitWidth) {
  // Sizes within a tight range: max=63 → bitWidth=6. Includes zeros that
  // the scatter step must filter.
  const std::vector<uint32_t> denseSizes = {1, 0, 63, 7, 0, 33, 0, 5};
  const auto buffer = buildLegacyFixedBitWidthTrailer(denseSizes);

  const auto* end = buffer.data() + buffer.size();
  auto [streamIndices, streamSizes] = readLegacyTrailerStreamMetadata(end);

  ASSERT_EQ(streamIndices.size(), 5u);
  EXPECT_EQ(streamIndices, (std::vector<uint32_t>{0, 2, 3, 5, 7}));
  EXPECT_EQ(streamSizes, (std::vector<uint32_t>{1, 63, 7, 33, 5}));
}

TEST(
    LegacyTrailerReaderTest,
    readLegacyTrailerStreamMetadataFixedBitWidthAllZeros) {
  // Regression: when every stream size is 0, master writes FixedBitWidth
  // with bitWidth=0 and no packed bytes (just the bitWidth byte and the
  // varint count). The decoder's sanity bound on count must respect
  // bitWidth — `count <= payloadSize * 8` is wrong because it assumes
  // each value takes >= 1 bit. The correct bound is
  // `count * bitWidth <= payloadSize * 8`, which trivially passes when
  // bitWidth == 0.
  const uint32_t count = 210;
  std::string buffer;
  buffer.push_back(static_cast<char>(EncodingType::FixedBitWidth));
  buffer.push_back(static_cast<char>(0)); // bitWidth = 0
  appendVarint32(buffer, count);
  // No packed bytes when bitWidth == 0.
  const uint32_t trailerSize = static_cast<uint32_t>(buffer.size());
  buffer.append(reinterpret_cast<const char*>(&trailerSize), sizeof(uint32_t));

  const auto* end = buffer.data() + buffer.size();
  auto [streamIndices, streamSizes] = readLegacyTrailerStreamMetadata(end);

  // All sizes are 0, so the scatter-to-sparse step filters everything out.
  EXPECT_TRUE(streamIndices.empty());
  EXPECT_TRUE(streamSizes.empty());
}

} // namespace
} // namespace facebook::nimble::serde::legacy
