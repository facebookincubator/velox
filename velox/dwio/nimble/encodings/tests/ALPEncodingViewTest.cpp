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

#include <limits>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/tests/EncodingLayoutTestHelper.h"

using namespace facebook;

class ALPEncodingViewTest : public nimble::test::EncodingViewTest {
 protected:
  // Encodes 'values' as ALP over Varint-encoded integers and checks every read
  // of a view over the result.
  void expectVarintEncodedReads(
      const nimble::Vector<float>& values,
      const std::vector<uint32_t>& positions) {
    const nimble::EncodingLayout layout{
        nimble::EncodingType::ALP,
        {},
        nimble::CompressionType::Uncompressed,
        {nimble::VarintEnc{}, nimble::TrivialEnc{}, nimble::TrivialEnc{}}};
    for (const auto useVarint : {false, true}) {
      SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
      const nimble::Encoding::Options options{
          .useVarintRowCount = useVarint,
          .cacheEncodingView = cacheEncodingView(),
      };
      const auto encoded = nimble::encodeFromCustomLayout<float>(
          layout,
          std::span<const float>{values.data(), values.size()},
          *buffer_,
          options);
      const char* pos = encoded.data() +
          nimble::EncodingPrefix::prefixSize(encoded, useVarint);
      if (nimble::detail::alp::readHeader(pos).hasExceptions) {
        nimble::varint::readVarint32(&pos);
      }
      const auto encodedValuesSize = nimble::varint::readVarint32(&pos);
      ASSERT_EQ(
          nimble::EncodingPrefix::encodingType({pos, encodedValuesSize}),
          nimble::EncodingType::Varint);
      expectEncodedReads(encoded, values, positions, options);
    }
  }
};

TEST_P(ALPEncodingViewTest, readsAlpEncoding) {
  expectReads<nimble::ALPEncoding<float>>(
      makeVector<float>({-12.5F, -1.25F, 0.0F, 1.25F, 12.5F}), {4, 0, 2, 1, 3});
  expectReads<nimble::ALPEncoding<double>>(
      makeVector<double>({-12.5, -1.25, 0.0, 1.25, 12.5}), {4, 0, 2, 1, 3});
  expectReads<nimble::ALPEncoding<float>>(
      makeVector<float>({1.25F, std::numeric_limits<float>::infinity(), 2.5F}),
      {1, 2, 0});
}

TEST_P(ALPEncodingViewTest, readsVarintEncodedValues) {
  expectVarintEncodedReads(
      randomAlpData<float>(/*seed=*/17), randomizedPositions(/*seed=*/18));
  // Infinity has no ALP integer, so it is stored as an exception.
  expectVarintEncodedReads(
      makeVector<float>(
          {1.25F, std::numeric_limits<float>::infinity(), 2.5F, -7.5F}),
      {3, 1, 0, 2, 1});
}

TEST_P(ALPEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/14);

  expectConcurrentReads<nimble::ALPEncoding<float>>(
      randomAlpData<float>(/*seed=*/15), positions);
  expectConcurrentReads<nimble::ALPEncoding<double>>(
      randomAlpData<double>(/*seed=*/16), positions);
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(ALPEncodingViewTest);
