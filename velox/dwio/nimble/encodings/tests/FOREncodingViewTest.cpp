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

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using FOREncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsForEncoding) {
  expectReads<nimble::ForEncoding<int32_t>>(
      makeVector({100, 101, 102, 103, 5000, 104, 105, 106, -7, -6, -5}),
      {8, 0, 4, 10, 5, 2});
  expectReads<nimble::ForEncoding<uint32_t>>(
      makeVector<uint32_t>({7, 8, 9, 10, 4096, 4097, 4098, 4099}),
      {7, 0, 4, 6, 2});
}

TEST_F(FOREncodingViewTest, readsInternallyCompressedPayload) {
  const auto values = randomPforData(/*seed=*/38);
  const auto positions = randomizedPositions(/*seed=*/39);
  for (const auto compressionType :
       {nimble::CompressionType::Zstd, nimble::CompressionType::MetaInternal}) {
    SCOPED_TRACE(nimble::toString(compressionType));
    expectReads<nimble::ForEncoding<uint32_t>>(
        values, positions, {}, compressionType);
  }
}

DEBUG_ONLY_TEST_F(FOREncodingViewTest, rejectsShortCompressedPayload) {
  constexpr uint32_t kRowCount = 1'016;
  nimble::Vector<uint32_t> values{pool_.get(), kRowCount};
  for (uint32_t i{0}; i < kRowCount; ++i) {
    values[i] = i % 2;
  }
  const auto serialized =
      nimble::test::Encoder<nimble::ForEncoding<uint32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Zstd);
  const auto prefixSize =
      nimble::EncodingPrefix::serializedSize(kRowCount, /*useVarint=*/false);
  ASSERT_EQ(
      static_cast<nimble::CompressionType>(serialized[prefixSize]),
      nimble::CompressionType::Zstd);

  std::string malformed{serialized};
  char* position = malformed.data();
  nimble::EncodingPrefix::serialize(
      nimble::EncodingType::FOR,
      nimble::DataType::Uint32,
      kRowCount + 1,
      /*useVarint=*/false,
      position);

  NIMBLE_ASSERT_THROW(
      nimble::createEncodingView(malformed, pool_.get(), {}),
      "FOR packed payload is shorter than required");
}

TEST_F(FOREncodingViewTest, concurrent) {
  expectConcurrentReads<nimble::ForEncoding<uint32_t>>(
      randomPforData(/*seed=*/31), randomizedPositions(/*seed=*/32));
}
