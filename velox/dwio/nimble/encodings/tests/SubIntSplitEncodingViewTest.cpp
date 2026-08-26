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
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
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
    for (const auto [offset, length] : std::vector<std::pair<uint32_t, uint32_t>>{
             {0, 0}, {kRows, 0}, {0, 1}, {5, 3}, {0, kRows}, {7, kRows - 7}}) {
      SCOPED_TRACE(fmt::format("offset={} length={}", offset, length));
      std::vector<T> actual(length);
      view->read(offset, length, actual.data());
      for (uint32_t i = 0; i < length; ++i) {
        ASSERT_EQ(actual[i], values[offset + i]) << "row " << (offset + i);
      }
    }
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

TEST_F(SubIntSplitEncodingViewTest, rejectsNonSubIntSplitStream) {
  const auto values = makeStructuredValues<uint64_t>(pool_.get(), 64);
  auto serialized = nimble::test::Encoder<nimble::TrivialEncoding<uint64_t>>::
      encode(*buffer_, values);
  // The factory dispatches on the stream's own type, so a Trivial stream must
  // not come back as a SubIntSplit view.
  auto view = nimble::createEncodingView(serialized, pool_.get());
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->encodingType(), nimble::EncodingType::Trivial);
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
  ASSERT_FALSE(nimble::supportsEncodingView(
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
