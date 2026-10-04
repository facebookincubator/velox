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

#include "velox/dwio/nimble/encodings/views/EncodingViewUtils.h"

#include <gtest/gtest.h>

#include <cstdint>

#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/EliasFanoEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

// The dispatch macro NIMBLE_FAST_VIEW_TYPES enumerates the encoding types
// whose readAt path is devirtualized. This test file covers a representative
// subset from that list (fast path) plus one viewable type NOT in the list
// (fallback), so both branches of the switch are exercised.
class EncodingViewUtilsTest : public nimble::test::EncodingViewTest {};

TEST_F(EncodingViewUtilsTest, readEncodingViewAtFastPathTrivial) {
  const nimble::Encoding::Options options;
  auto values = randomInt32(/*seed=*/1);
  auto serialized =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Uncompressed, options);
  auto view = nimble::createEncodingView(serialized, pool_.get(), options);
  ASSERT_NE(view, nullptr);
  ASSERT_EQ(view->encodingType(), nimble::EncodingType::Trivial);
  for (uint32_t i = 0; i < values.size(); i += 37) {
    EXPECT_EQ(nimble::readEncodingViewAt<int32_t>(view.get(), i), values[i])
        << "i=" << i;
  }
}

TEST_F(EncodingViewUtilsTest, readEncodingViewAtFastPathFixedBitWidth) {
  const nimble::Encoding::Options options;
  auto values = randomNarrowUnsigned<uint32_t>(/*seed=*/2);
  auto serialized =
      nimble::test::Encoder<nimble::FixedBitWidthEncoding<uint32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Uncompressed, options);
  auto view = nimble::createEncodingView(serialized, pool_.get(), options);
  ASSERT_NE(view, nullptr);
  ASSERT_EQ(view->encodingType(), nimble::EncodingType::FixedBitWidth);
  for (uint32_t i = 0; i < values.size(); i += 29) {
    EXPECT_EQ(nimble::readEncodingViewAt<uint32_t>(view.get(), i), values[i])
        << "i=" << i;
  }
}

TEST_F(EncodingViewUtilsTest, readEncodingViewAtFastPathRLE) {
  const nimble::Encoding::Options options;
  auto values = randomRleInt32(/*seed=*/3);
  auto serialized = nimble::test::Encoder<nimble::RLEEncoding<int32_t>>::encode(
      *buffer_, values, nimble::CompressionType::Uncompressed, options);
  auto view = nimble::createEncodingView(serialized, pool_.get(), options);
  ASSERT_NE(view, nullptr);
  ASSERT_EQ(view->encodingType(), nimble::EncodingType::RLE);
  for (uint32_t i = 0; i < values.size(); i += 13) {
    EXPECT_EQ(nimble::readEncodingViewAt<int32_t>(view.get(), i), values[i])
        << "i=" << i;
  }
}

TEST_F(EncodingViewUtilsTest, readEncodingViewAtFastPathConstant) {
  const nimble::Encoding::Options options;
  auto values = constantInt32(/*value=*/42);
  auto serialized =
      nimble::test::Encoder<nimble::ConstantEncoding<int32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Uncompressed, options);
  auto view = nimble::createEncodingView(serialized, pool_.get(), options);
  ASSERT_NE(view, nullptr);
  ASSERT_EQ(view->encodingType(), nimble::EncodingType::Constant);
  for (uint32_t i = 0; i < values.size(); i += 51) {
    EXPECT_EQ(nimble::readEncodingViewAt<int32_t>(view.get(), i), 42)
        << "i=" << i;
  }
}

TEST_F(EncodingViewUtilsTest, readEncodingViewAtFastPathPFOR) {
  const nimble::Encoding::Options options;
  auto values = randomPforData(/*seed=*/4);
  auto serialized =
      nimble::test::Encoder<nimble::PFOREncoding<uint32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Uncompressed, options);
  auto view = nimble::createEncodingView(serialized, pool_.get(), options);
  ASSERT_NE(view, nullptr);
  ASSERT_EQ(view->encodingType(), nimble::EncodingType::PFOR);
  for (uint32_t i = 0; i < values.size(); i += 23) {
    EXPECT_EQ(nimble::readEncodingViewAt<uint32_t>(view.get(), i), values[i])
        << "i=" << i;
  }
}

// EliasFano is viewable but intentionally NOT in NIMBLE_FAST_VIEW_TYPES, so
// this exercises the default fallback branch that goes through the base-class
// vtable `readAt(index, void*)` path. Correctness must match the fast path's
// contract regardless.
TEST_F(EncodingViewUtilsTest, readEncodingViewAtFallbackEliasFano) {
  const nimble::Encoding::Options options;
  // EliasFano requires strictly ascending unsigned integers.
  nimble::Vector<uint32_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  uint32_t v = 0;
  for (uint32_t i = 0; i < kConcurrentRows; ++i) {
    v += 1 + (i % 5);
    values.push_back(v);
  }
  auto serialized =
      nimble::test::Encoder<nimble::EliasFanoEncoding<uint32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Uncompressed, options);
  auto view = nimble::createEncodingView(serialized, pool_.get(), options);
  ASSERT_NE(view, nullptr);
  ASSERT_EQ(view->encodingType(), nimble::EncodingType::EliasFano);
  // Sanity-check that EliasFano really goes through the fallback branch --
  // if a future refactor puts it on NIMBLE_FAST_VIEW_TYPES, correctness stays
  // but this test stops covering the fallback; add another off-list type
  // (BitRangeSplit, SparseBool for bool, ALP for floats) to keep coverage.
  for (uint32_t i = 0; i < values.size(); i += 41) {
    EXPECT_EQ(nimble::readEncodingViewAt<uint32_t>(view.get(), i), values[i])
        << "i=" << i;
  }
}
