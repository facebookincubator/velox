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

#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/tests/EncodingLayoutTestHelper.h"

using namespace facebook;

class DictionaryEncodingViewTest : public nimble::test::EncodingViewTest {
 protected:
  // Encodes 'values' with 'layout', checks that the alphabet was written with
  // 'alphabetEncoding', and checks every read of a view over the result.
  template <typename T>
  void expectLayoutReads(
      const nimble::EncodingLayout& layout,
      nimble::EncodingType alphabetEncoding,
      const nimble::Vector<T>& values,
      const std::vector<uint32_t>& positions) {
    for (const auto useVarint : {false, true}) {
      SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
      const nimble::Encoding::Options options{
          .useVarintRowCount = useVarint,
          .cacheEncodingView = cacheEncodingView(),
      };
      const auto encoded = nimble::encodeFromCustomLayout<T>(
          layout,
          std::span<const T>{values.data(), values.size()},
          *buffer_,
          options);
      const char* pos = encoded.data() +
          nimble::EncodingPrefix::prefixSize(encoded, useVarint);
      const auto alphabetSize = nimble::encoding::readUint32(pos);
      ASSERT_EQ(
          nimble::EncodingPrefix::encodingType({pos, alphabetSize}),
          alphabetEncoding);
      expectEncodedReads(encoded, values, positions, options);
    }
  }
};

TEST_P(DictionaryEncodingViewTest, readsDictionaryEncoding) {
  expectReads<nimble::DictionaryEncoding<int32_t>>(
      makeVector({5, 7, 5, 9, 7, 5, 11}), {6, 0, 3, 2, 5});
}

TEST_P(
    DictionaryEncodingViewTest,
    readsDictionaryEncodingForSupportedTypeFamilies) {
  expectReads<nimble::DictionaryEncoding<uint32_t>>(
      makeVector<uint32_t>({5, 7, 5, 9, 7, 5, 11}), {6, 0, 3, 2, 5});
  expectReads<nimble::DictionaryEncoding<float>>(
      makeVector<float>({1.25F, 2.5F, 1.25F, -3.75F, 2.5F}), {4, 0, 3, 2, 1});
  expectReads<nimble::DictionaryEncoding<double>>(
      makeVector<double>({1.25, 2.5, 1.25, -3.75, 2.5}), {4, 0, 3, 2, 1});
  expectReads<nimble::DictionaryEncoding<std::string_view>>(
      makeVector<std::string_view>({"alpha", "beta", "alpha", "gamma", "beta"}),
      {4, 0, 3, 2, 1});
}

TEST_P(DictionaryEncodingViewTest, readsBitPackedAlphabet) {
  expectLayoutReads(
      nimble::DictionaryEnc{
          .alphabet = nimble::FixedBitWidthEnc{},
          .indices = nimble::FixedBitWidthEnc{},
      },
      nimble::EncodingType::FixedBitWidth,
      randomDictionaryUint32(/*seed=*/11),
      randomizedPositions(/*seed=*/12));
}

TEST_P(DictionaryEncodingViewTest, readsAlpAlphabetOverVarint) {
  // The writer picks this layout for vectors of small integers when ALP is
  // allowed. Every row read reads the alphabet, so its Varint view is read
  // once per row.
  expectLayoutReads(
      nimble::DictionaryEnc{
          .alphabet =
              nimble::EncodingLayout{
                  nimble::EncodingType::ALP,
                  {},
                  nimble::CompressionType::Uncompressed,
                  {nimble::VarintEnc{},
                   nimble::TrivialEnc{},
                   nimble::TrivialEnc{}}},
          .indices = nimble::FixedBitWidthEnc{},
      },
      nimble::EncodingType::ALP,
      randomAlpData<float>(/*seed=*/13),
      randomizedPositions(/*seed=*/14));
}

TEST_P(DictionaryEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/8);
  std::vector<std::string> backing;

  expectConcurrentReads<nimble::DictionaryEncoding<uint32_t>>(
      randomDictionaryUint32(/*seed=*/9), positions);
  expectConcurrentReads<nimble::DictionaryEncoding<std::string_view>>(
      randomStringViews(backing, /*seed=*/10), positions);
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(DictionaryEncodingViewTest);
