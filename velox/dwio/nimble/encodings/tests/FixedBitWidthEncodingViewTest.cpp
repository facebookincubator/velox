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

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <utility>

#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"

using namespace facebook;

namespace {

// The lane width and baseline of a FixedBitWidth chunk of values of type T,
// encoded without a varint row count. The lane width is the bit width when the
// payload is typed lanes (whole lanes and 8 zero bytes), and 0 when it is
// bit-packed (FixedBitArray::bufferSize() bytes).
template <typename T>
std::pair<int, uint64_t> laneBitsAndBaseline(
    std::string_view encoded,
    velox::memory::MemoryPool& pool) {
  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;
  const auto rowCount =
      nimble::EncodingPrefix::readRowCount(encoded, /*useVarint=*/false);
  const char* pos = encoded.data() + nimble::EncodingPrefix::kFixedPrefixSize;
  const auto compressionType = static_cast<nimble::CompressionType>(*pos++);
  PhysicalType baseline;
  std::memcpy(&baseline, pos, sizeof(baseline));
  pos += sizeof(baseline);
  const int bitWidth = static_cast<uint8_t>(*pos++);
  const std::string_view stored{
      pos, static_cast<size_t>(encoded.data() + encoded.size() - pos)};
  auto payloadSize = stored.size();
  if (compressionType != nimble::CompressionType::Uncompressed) {
    payloadSize = nimble::Compression::uncompress(
                      pool,
                      compressionType,
                      nimble::DataType::Undefined,
                      stored,
                      /*decompressCounter=*/nullptr)
                      ->size();
  }
  if (payloadSize == nimble::FixedBitArray::bufferSize(rowCount, bitWidth)) {
    return {0, baseline};
  }
  EXPECT_EQ(
      payloadSize,
      nimble::FixedBitWidthTypedLanes::payloadSize(rowCount, bitWidth));
  return {bitWidth, baseline};
}

} // namespace

using FixedBitWidthEncodingViewTest = nimble::test::EncodingViewTest;

TEST_P(FixedBitWidthEncodingViewTest, readsFixedBitWidthEncoding) {
  expectReads<nimble::FixedBitWidthEncoding<int32_t>>(
      makeVector({10, 12, 15, 31, 33, 63}), {5, 0, 4, 2, 1});
}

TEST_P(FixedBitWidthEncodingViewTest, readsInternallyCompressedPayload) {
  const auto values = randomNarrowUnsigned<uint32_t>(/*seed=*/36);
  const auto positions = randomizedPositions(/*seed=*/37);
  for (const auto compressionType :
       {nimble::CompressionType::Zstd, nimble::CompressionType::MetaInternal}) {
    SCOPED_TRACE(nimble::toString(compressionType));
    expectReads<nimble::FixedBitWidthEncoding<uint32_t>>(
        values, positions, {}, compressionType);
  }
}

// With typed lanes on (floor 0) and OpenZL, which reads the payload's data
// type, chunks of 16 or more bits are stored as typed lanes, and the view
// reads them back exactly: 16-bit narrow lanes (uint32 over 12 bits), 32-bit
// narrow lanes (uint64 over 20 bits) and full-width lanes, which store the
// values with baseline 0 (uint32 over 20 bits).
TEST_P(FixedBitWidthEncodingViewTest, readsTypedLanePayload) {
  nimble::Encoding::Options options;
  options.fixedBitWidthTypedLanes = true;
  options.fixedBitWidthTypedLanesMinValues = 0;
  const auto positions = randomizedPositions(/*seed=*/38);
  const auto check = [&]<typename T>(
                         uint32_t seed,
                         uint64_t base,
                         int bits,
                         int expectedLaneBits,
                         bool fullWidth) {
    SCOPED_TRACE(fmt::format("sizeof {}, {} bits", sizeof(T), bits));
    std::mt19937 rng{seed};
    std::uniform_int_distribution<uint64_t> offset{
        0, (uint64_t{1} << bits) - 1};
    nimble::Vector<T> values{pool_.get()};
    values.reserve(kConcurrentRows);
    for (uint32_t i = 0; i < kConcurrentRows; ++i) {
      values.push_back(static_cast<T>(base + offset(rng)));
    }
    // Both ends of the range, so the chunk needs exactly 'bits' bits.
    values[0] = static_cast<T>(base);
    values[1] = static_cast<T>(base + (uint64_t{1} << bits) - 1);

    const auto encoded =
        nimble::test::Encoder<nimble::FixedBitWidthEncoding<T>>::encode(
            *buffer_, values, nimble::CompressionType::OpenZL, options);
    const auto [laneBits, baseline] = laneBitsAndBaseline<T>(encoded, *pool_);
    EXPECT_EQ(laneBits, expectedLaneBits);
    EXPECT_EQ(baseline, fullWidth ? 0 : base);
    expectReads<nimble::FixedBitWidthEncoding<T>>(
        values, positions, options, nimble::CompressionType::OpenZL);
  };
  check.operator()<uint32_t>(/*seed=*/39, 1'000, 12, 16, /*fullWidth=*/false);
  check.operator()<uint64_t>(
      /*seed=*/40, 1'000'000, 20, 32, /*fullWidth=*/false);
  check.operator()<uint32_t>(/*seed=*/41, 7, 20, 32, /*fullWidth=*/true);
}

TEST_P(FixedBitWidthEncodingViewTest, concurrent) {
  expectConcurrentReads<nimble::FixedBitWidthEncoding<uint32_t>>(
      randomNarrowUnsigned<uint32_t>(/*seed=*/6),
      randomizedPositions(/*seed=*/7));
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(FixedBitWidthEncodingViewTest);
