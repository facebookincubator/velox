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

#include <bit>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/SubIntSplitEncodingView.h"

using namespace facebook;

namespace {

// Values shaped like the structured IDs SubIntSplit exists for: a wide constant
// prefix, a mid field that changes slowly, and a low counter. A split over this
// produces several sections of differing width and differing sub-encoding,
// which is what the view has to reassemble.
template <typename T>
nimble::Vector<T> makeStructuredValues(
    velox::memory::MemoryPool* pool,
    uint32_t count) {
  using U = std::make_unsigned_t<typename nimble::TypeTraits<T>::physicalType>;
  nimble::Vector<T> values{pool};
  values.reserve(count);
  const U prefix = sizeof(U) == 4 ? static_cast<U>(0x12340000u)
                                  : static_cast<U>(0x1234567890000000ULL);
  for (uint32_t i = 0; i < count; ++i) {
    const U mid = static_cast<U>((i / 64) & 0x3F) << 10;
    const U low = static_cast<U>(i & 0x3FF);
    values.push_back(static_cast<T>(prefix | mid | low));
  }
  return values;
}

std::vector<uint32_t> probePositions(uint32_t count) {
  std::vector<uint32_t> positions;
  for (uint32_t i = 0; i < count; i += 37) {
    positions.push_back(i);
  }
  positions.push_back(0);
  positions.push_back(count - 1);
  return positions;
}

} // namespace

class SubIntSplitEncodingViewTest : public nimble::test::EncodingViewTest {
 protected:
  static constexpr uint32_t kRows = 1024;

  // Encodes through the real nested selection policy, so each section gets
  // whatever sub-encoding the policy actually picks, then reads every row back
  // through the view both one at a time and as a range.
  template <typename T>
  void expectViewMatches(nimble::CompressionType compressionType) {
    SCOPED_TRACE(fmt::format("compression={}", compressionType));
    const auto values = makeStructuredValues<T>(pool_.get(), kRows);
    const nimble::Encoding::Options options{};
    auto serialized =
        nimble::test::Encoder<nimble::SubIntSplitEncoding<T>>::encode(
            *buffer_,
            values,
            compressionType,
            options,
            /*realNestedSelection=*/true);

    auto view = nimble::createEncodingView(serialized, pool_.get(), options);
    ASSERT_NE(view, nullptr);
    ASSERT_EQ(view->encodingType(), nimble::EncodingType::SubIntSplit);
    ASSERT_EQ(view->rowCount(), kRows);

    for (const auto position : probePositions(kRows)) {
      SCOPED_TRACE(fmt::format("position={}", position));
      T value{};
      view->readAt(position, &value);
      EXPECT_EQ(value, values[position]);
    }

    // Ranges, including one that spans more than the view's internal chunk so
    // the chunked path is exercised rather than only its first iteration.
    for (const auto [offset, length] :
         std::vector<std::pair<uint32_t, uint32_t>>{
             {0, 0}, {kRows, 0}, {0, 1}, {5, 3}, {0, kRows}, {7, kRows - 7}}) {
      SCOPED_TRACE(fmt::format("offset={} length={}", offset, length));
      std::vector<T> actual(length);
      view->read(offset, length, actual.data());
      for (uint32_t i = 0; i < length; ++i) {
        ASSERT_EQ(actual[i], values[offset + i]) << "row " << (offset + i);
      }
    }
  }

  // Long enough that a bridged group reaches the view's 1024-row chunk cap
  // several times over and still leaves a ragged last chunk.
  template <typename T>
  void expectRangeListsMatch(nimble::CompressionType compressionType) {
    SCOPED_TRACE(fmt::format("compression={}", compressionType));
    const auto values = makeStructuredValues<T>(pool_.get(), 20'011);
    auto serialized =
        nimble::test::Encoder<nimble::SubIntSplitEncoding<T>>::encode(
            *buffer_,
            values,
            compressionType,
            nimble::Encoding::Options{},
            /*realNestedSelection=*/true);
    auto view = nimble::createEncodingView(serialized, pool_.get(), {});
    ASSERT_NE(view, nullptr);
    ASSERT_EQ(view->encodingType(), nimble::EncodingType::SubIntSplit);
    nimble::test::expectRangeListReads(*view, values);
  }
};

TEST_F(SubIntSplitEncodingViewTest, readsEveryWidth) {
  expectViewMatches<int32_t>(nimble::CompressionType::Uncompressed);
  expectViewMatches<uint32_t>(nimble::CompressionType::Uncompressed);
  expectViewMatches<int64_t>(nimble::CompressionType::Uncompressed);
  expectViewMatches<uint64_t>(nimble::CompressionType::Uncompressed);
}

// A compressed sub-stream has no view of its own, so the section falls back to
// being decoded once into an array. Indexed access has to survive that, which
// is the whole reason the fallback decodes rather than keeping a cursor.
TEST_F(SubIntSplitEncodingViewTest, readsWithCompressedSubStreams) {
  expectViewMatches<uint64_t>(nimble::CompressionType::Zstd);
  expectViewMatches<int64_t>(nimble::CompressionType::Zstd);
}

// A range-list read bridges short gaps between ranges with one staged bulk
// decode, so its rows reach the output by a different path from a per-range
// loop: every list shape has to come back exactly as a loop would return it,
// including over compressed sub-streams, whose sections are materialized.
TEST_F(SubIntSplitEncodingViewTest, readsRangeLists) {
  for (const auto compressionType :
       {nimble::CompressionType::Uncompressed, nimble::CompressionType::Zstd}) {
    expectRangeListsMatch<int32_t>(compressionType);
    expectRangeListsMatch<uint32_t>(compressionType);
    expectRangeListsMatch<int64_t>(compressionType);
    expectRangeListsMatch<uint64_t>(compressionType);
  }
}

// A stream stored as its distance from a line through the rows has to add the
// line back on every view read path: point, range, arbitrary indices and range
// lists, with and without section transforms layered over the residuals.
TEST_F(SubIntSplitEncodingViewTest, readsThroughRowFrame) {
  constexpr uint32_t kFrameRows = 40'000;
  std::mt19937 generator{5};
  nimble::Vector<uint64_t> values{pool_.get()};
  values.reserve(kFrameRows);
  for (uint32_t row = 0; row < kFrameRows; ++row) {
    // A tag that ignores the rows, a counter, and a field that tracks the
    // counter to within eight either way.
    const uint64_t tag = generator() % 12;
    const uint64_t tracking = row < 8 ? row : row + generator() % 16 - 8;
    values.push_back((tag << 56) | (uint64_t{row} << 28) | tracking);
  }

  for (const bool autoTransform : {false, true}) {
    for (const auto compressionType :
         {nimble::CompressionType::Uncompressed,
          nimble::CompressionType::Zstd}) {
      SCOPED_TRACE(
          fmt::format(
              "autoTransform={} compression={}",
              autoTransform,
              compressionType));
      nimble::Encoding::Options options;
      options.subIntSplitAutoTransform = autoTransform;
      auto serialized =
          nimble::test::Encoder<nimble::SubIntSplitEncoding<uint64_t>>::encode(
              *buffer_,
              values,
              compressionType,
              options,
              /*realNestedSelection=*/true);
      nimble::subintsplit::RowFrame frame;
      nimble::subintsplit::parseSections(
          serialized, nimble::Encoding::kPrefixSize, nullptr, &frame);
      ASSERT_EQ(frame.slope, (uint64_t{1} << 28) + 1);

      auto view = nimble::createEncodingView(serialized, pool_.get(), options);
      ASSERT_NE(view, nullptr);
      ASSERT_EQ(view->rowCount(), kFrameRows);
      for (const auto position : probePositions(kFrameRows)) {
        uint64_t value{0};
        view->readAt(position, &value);
        ASSERT_EQ(value, values[position]) << "row " << position;
      }
      const std::vector<uint32_t> indices{39'999, 3, 4, 5, 5, 17'000, 0};
      std::vector<uint64_t> gathered(indices.size());
      view->readAt(indices, gathered.data());
      for (size_t i = 0; i < indices.size(); ++i) {
        ASSERT_EQ(gathered[i], values[indices[i]]) << "index " << indices[i];
      }
      for (const auto [offset, length] :
           std::vector<std::pair<uint32_t, uint32_t>>{
               {0, 1}, {1'023, 5'000}, {0, kFrameRows}, {39'990, 10}}) {
        std::vector<uint64_t> actual(length);
        view->read(offset, length, actual.data());
        for (uint32_t i = 0; i < length; ++i) {
          ASSERT_EQ(actual[i], values[offset + i]) << "row " << (offset + i);
        }
      }
      nimble::test::expectRangeListReads(*view, values);
    }
  }
}

TEST_F(SubIntSplitEncodingViewTest, rejectsNonSubIntSplitStream) {
  const auto values = makeStructuredValues<uint64_t>(pool_.get(), 64);
  auto serialized =
      nimble::test::Encoder<nimble::TrivialEncoding<uint64_t>>::encode(
          *buffer_, values);
  // The factory dispatches on the stream's own type, so a Trivial stream must
  // not come back as a SubIntSplit view.
  auto view = nimble::createEncodingView(serialized, pool_.get());
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->encodingType(), nimble::EncodingType::Trivial);
}

// The shared harness, as every other view test uses it. randomizedPositions and
// expectConcurrentReads are what pin the view's thread safety, which is why its
// chunk scratch is a stack buffer rather than a member.
TEST_F(SubIntSplitEncodingViewTest, sharedHarness) {
  const auto positions = probePositions(kRows);

  expectReads<nimble::SubIntSplitEncoding<int32_t>>(
      makeStructuredValues<int32_t>(pool_.get(), kRows),
      positions,
      /*baseOptions=*/{},
      nimble::CompressionType::Uncompressed,
      /*realNestedSelection=*/true);
  expectReads<nimble::SubIntSplitEncoding<uint64_t>>(
      makeStructuredValues<uint64_t>(pool_.get(), kRows),
      positions,
      /*baseOptions=*/{},
      nimble::CompressionType::Uncompressed,
      /*realNestedSelection=*/true);
}

TEST_F(SubIntSplitEncodingViewTest, concurrent) {
  const auto positions = probePositions(kRows);

  expectConcurrentReads<nimble::SubIntSplitEncoding<uint64_t>>(
      makeStructuredValues<uint64_t>(pool_.get(), kRows),
      positions,
      /*baseOptions=*/{},
      /*realNestedSelection=*/true);
  expectConcurrentReads<nimble::SubIntSplitEncoding<int32_t>>(
      makeStructuredValues<int32_t>(pool_.get(), kRows),
      positions,
      /*baseOptions=*/{},
      /*realNestedSelection=*/true);
}

// SubIntSplitEncodingTest covers six types; the view covered four. Float and
// double reach the view because TypeTraits<float>::physicalType is uint32_t, so
// the bit work is unsigned integer work either way.
TEST_F(SubIntSplitEncodingViewTest, readsFloatingPointTypes) {
  expectViewMatches<float>(nimble::CompressionType::Uncompressed);
  expectViewMatches<double>(nimble::CompressionType::Uncompressed);
}

// A nullable column whose non-null values SubIntSplit stores: the Nullable
// wrapper carries the nulls and the split carries the values, and both have to
// come back through the encoding and through its view.
TEST_F(SubIntSplitEncodingViewTest, nullableColumnRoundTrips) {
  constexpr uint32_t kNullableRows = 2'000;
  nimble::Vector<int64_t> nonNullValues{pool_.get()};
  nimble::Vector<bool> isNonNull{pool_.get(), kNullableRows};
  std::vector<std::optional<int64_t>> expected(kNullableRows);
  for (uint32_t row = 0; row < kNullableRows; ++row) {
    isNonNull[row] = row % 7 != 0;
    if (isNonNull[row]) {
      const auto value =
          static_cast<int64_t>((uint64_t{row} << 20) | (row % 13));
      nonNullValues.push_back(value);
      expected[row] = value;
    }
  }
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<int64_t>>(
          std::vector<std::pair<nimble::EncodingType, float>>{
              {nimble::EncodingType::SubIntSplit, 1.0}},
          nimble::CompressionOptions{},
          std::nullopt);
  const auto encoded = nimble::EncodingFactory::encodeNullable<int64_t>(
      std::move(policy), nonNullValues, isNonNull, *buffer_);

  auto encoding = nimble::EncodingFactory().create(
      *pool_, encoded, [](uint32_t) -> void* { return nullptr; });
  ASSERT_EQ(encoding->encodingType(), nimble::EncodingType::Nullable);
  ASSERT_NE(encoding->debugString(0).find("SubIntSplit"), std::string::npos);

  // Through the encoding.
  std::vector<int64_t> values(kNullableRows);
  std::vector<uint64_t> nonNullBits(
      velox::bits::nwords(kNullableRows), ~uint64_t{0});
  const auto numNonNulls = encoding->materializeNullable(
      kNullableRows, values.data(), [&]() -> void* {
        return nonNullBits.data();
      });
  EXPECT_EQ(numNonNulls, nonNullValues.size());
  for (uint32_t row = 0; row < kNullableRows; ++row) {
    SCOPED_TRACE(fmt::format("row={}", row));
    EXPECT_EQ(
        velox::bits::isBitSet(nonNullBits.data(), row),
        expected[row].has_value());
    if (expected[row].has_value()) {
      EXPECT_EQ(values[row], *expected[row]);
    }
  }

  // Through the view.
  auto view = nimble::createEncodingView(encoded, pool_.get());
  ASSERT_NE(view, nullptr);
  std::vector<uint32_t> indices(kNullableRows);
  std::iota(indices.begin(), indices.end(), 0);
  std::vector<int64_t> viewValues(kNullableRows);
  std::vector<uint32_t> nullIndices;
  const auto viewNonNulls = view->read(
      indices,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      viewValues.data());
  EXPECT_EQ(viewNonNulls + nullIndices.size(), kNullableRows);
  std::vector<uint32_t> expectedNulls;
  for (uint32_t row = 0; row < kNullableRows; ++row) {
    if (!expected[row].has_value()) {
      expectedNulls.push_back(row);
    } else {
      EXPECT_EQ(viewValues[row], *expected[row]) << "row=" << row;
    }
  }
  EXPECT_EQ(nullIndices, expectedNulls);
}

// Constant sections are folded into a seed at construction and dropped from the
// per-row loop, so both the folding and the degenerate case where nothing is
// left to loop over need covering.
TEST_F(SubIntSplitEncodingViewTest, readsWhenEverySectionIsConstant) {
  nimble::Vector<uint64_t> values{pool_.get()};
  for (uint32_t i = 0; i < 512; ++i) {
    values.push_back(0x1234567890ABCDEFULL);
  }
  auto serialized =
      nimble::test::Encoder<nimble::SubIntSplitEncoding<uint64_t>>::encode(
          *buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          {},
          /*realNestedSelection=*/true);

  auto view = nimble::createEncodingView(serialized, pool_.get(), {});
  ASSERT_NE(view, nullptr);
  ASSERT_EQ(view->rowCount(), values.size());

  uint64_t value{};
  view->readAt(0, &value);
  EXPECT_EQ(value, values[0]);
  view->readAt(511, &value);
  EXPECT_EQ(value, values[511]);

  std::vector<uint64_t> range(values.size());
  view->read(0, values.size(), range.data());
  for (uint32_t i = 0; i < values.size(); ++i) {
    ASSERT_EQ(range[i], values[i]) << "row " << i;
  }

  // A range that spans more than one internal chunk, so the seed is applied per
  // chunk rather than only on the first.
  std::vector<uint64_t> tail(400);
  view->read(100, 400, tail.data());
  for (uint32_t i = 0; i < 400; ++i) {
    ASSERT_EQ(tail[i], values[100 + i]) << "row " << (100 + i);
  }
}

// Settles why makeSectionView cannot use a shallow "is this stream compressed"
// predicate. An RLE stream carries no compression byte of its own; its run
// values do, nested one level down, so a whitelist keyed on the outer encoding
// type would call this viewable and construction would then throw.
TEST_F(SubIntSplitEncodingViewTest, compressionNestsBelowTheOuterEncoding) {
  nimble::Vector<uint32_t> values{pool_.get()};
  for (uint32_t run = 0; run < 64; ++run) {
    for (uint32_t i = 0; i < 32; ++i) {
      values.push_back(run);
    }
  }
  auto serialized =
      nimble::test::Encoder<nimble::RLEEncoding<uint32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Zstd);

  // The outer stream is RLE, which supportsEncodingView() reports as viewable.
  ASSERT_EQ(
      nimble::EncodingPrefix::encodingType(serialized),
      nimble::EncodingType::RLE);
  ASSERT_TRUE(
      nimble::supportsEncodingView(
          nimble::EncodingPrefix::encodingType(serialized)));

  // Whether construction succeeds depends on what the nested values stream did
  // with the codec, which is exactly what an outer predicate cannot see. Assert
  // only that both outcomes are handled, never that it silently misbehaves.
  bool threw = false;
  try {
    auto view = nimble::createEncodingView(serialized, pool_.get(), {});
    ASSERT_NE(view, nullptr);
    uint32_t value{};
    view->readAt(100, &value);
    EXPECT_EQ(value, values[100]);
  } catch (const nimble::NimbleException&) {
    threw = true;
  }
  RecordProperty("nestedCompressionThrows", threw ? "yes" : "no");

  // Whatever the view does, the SubIntSplit fallback must serve the stream.
  nimble::detail::MaterializedEncodingView<uint32_t> fallback{
      serialized, pool_.get(), {}};
  ASSERT_EQ(fallback.rowCount(), values.size());
  uint32_t value{};
  fallback.readAt(100, &value);
  EXPECT_EQ(value, values[100]);
}

// The fallback in isolation. Varint is the one encoding in the default nested
// selection inventory with no view, so it is the case that reaches
// MaterializedEncodingView on an uncompressed column.
TEST_F(SubIntSplitEncodingViewTest, materializedFallbackServesVarint) {
  nimble::Vector<uint32_t> values{pool_.get()};
  for (uint32_t i = 0; i < 512; ++i) {
    values.push_back(i * 7919u);
  }
  auto serialized =
      nimble::test::Encoder<nimble::VarintEncoding<uint32_t>>::encode(
          *buffer_, values);
  ASSERT_FALSE(
      nimble::supportsEncodingView(
          nimble::EncodingPrefix::encodingType(serialized)));

  nimble::detail::MaterializedEncodingView<uint32_t> view{
      serialized, pool_.get(), {}};
  ASSERT_EQ(view.rowCount(), values.size());
  for (uint32_t i = 0; i < values.size(); ++i) {
    uint32_t value{};
    view.readAt(i, &value);
    ASSERT_EQ(value, values[i]) << "row " << i;
  }

  std::vector<uint32_t> range(100);
  view.read(11, 100, range.data());
  for (uint32_t i = 0; i < 100; ++i) {
    ASSERT_EQ(range[i], values[11 + i]) << "row " << (11 + i);
  }
}
