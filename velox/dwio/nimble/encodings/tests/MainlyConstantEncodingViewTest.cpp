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

#include <algorithm>
#include <numeric>
#include <vector>

#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"

using namespace facebook;

using MainlyConstantEncodingViewTest = nimble::test::EncodingViewTest;

TEST_P(MainlyConstantEncodingViewTest, readsMainlyConstantEncoding) {
  expectReads<nimble::MainlyConstantEncoding<int32_t>>(
      makeVector({7, 7, 11, 7, 7, 19, 7, 23, 7}), {8, 0, 2, 5, 7, 3});
  expectReads<nimble::MainlyConstantEncoding<float>>(
      makeVector<float>({1.25F, 1.25F, 2.5F, 1.25F, -3.75F, 1.25F}),
      {5, 0, 2, 4, 1});
  expectReads<nimble::MainlyConstantEncoding<double>>(
      makeVector<double>({1.25, 1.25, 2.5, 1.25, -3.75, 1.25}),
      {5, 0, 2, 4, 1});
  expectReads<nimble::MainlyConstantEncoding<std::string_view>>(
      makeVector<std::string_view>(
          {"alpha", "alpha", "beta", "alpha", "gamma", "alpha"}),
      {5, 0, 2, 4, 1});
}

TEST_P(MainlyConstantEncodingViewTest, readsAcrossWordBoundaries) {
  struct Pattern {
    const char* name;
    bool (*isOther)(uint32_t row);
  };
  const std::vector<Pattern> patterns = {
      {"sparse",
       [](uint32_t row) {
         return row % 64 == 0 || row % 64 == 63 || row % 11 == 4;
       }},
      {"runs", [](uint32_t row) { return (row / 100) % 2 == 1; }},
      {"alternating", [](uint32_t row) { return row % 2 == 1; }},
  };

  for (const auto& pattern : patterns) {
    for (const uint32_t numRows : {2U, 63U, 64U, 65U, 128U, 1'000U}) {
      SCOPED_TRACE(
          fmt::format("pattern={}, numRows={}", pattern.name, numRows));
      nimble::Vector<int32_t> values{pool_.get()};
      for (uint32_t row = 0; row < numRows; ++row) {
        const auto isOther = pattern.isOther(row) || row == numRows - 1;
        values.push_back(isOther ? static_cast<int32_t>(1'000 + row) : 7);
      }
      nimble::Encoding::Options options;
      options.cacheEncodingView = cacheEncodingView();
      const auto serialized = nimble::test::
          Encoder<nimble::MainlyConstantEncoding<int32_t>>::encode(
              *buffer_, values, nimble::CompressionType::Uncompressed, options);
      const auto view =
          nimble::createEncodingView(serialized, pool_.get(), options);
      ASSERT_NE(view, nullptr);

      for (const uint32_t offset :
           {0U, 1U, 31U, 63U, 64U, 65U, numRows - 1, numRows}) {
        if (offset > numRows) {
          continue;
        }
        for (const uint32_t length :
             {0U, 1U, 2U, 63U, 64U, 65U, 130U, numRows}) {
          if (length > numRows - offset) {
            continue;
          }
          expectRangeRead(*view, values, offset, length);
        }
      }

      std::vector<uint32_t> positions(numRows);
      std::iota(positions.begin(), positions.end(), 0);
      expectIndexedRead(*view, values, positions);
      expectSelectedRead(*view, values, positions);
      std::reverse(positions.begin(), positions.end());
      expectIndexedRead(*view, values, positions);
    }
  }
}

TEST_P(MainlyConstantEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/29);

  auto values = constantInt32(5);
  for (uint32_t i = 0; i < values.size(); i += 17) {
    values[i] = static_cast<int32_t>(i);
  }
  expectConcurrentReads<nimble::MainlyConstantEncoding<int32_t>>(
      values, positions);
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(MainlyConstantEncodingViewTest);
