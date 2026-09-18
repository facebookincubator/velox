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

#include <array>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble::test {
namespace {

class NullableEncodingViewTest : public EncodingViewTest {
 protected:
  std::unique_ptr<EncodingView> makeView(
      const std::vector<std::optional<int64_t>>& rows) {
    Vector<int64_t> nonNullValues{pool_.get()};
    Vector<bool> isNonNull{pool_.get(), rows.size()};
    for (size_t i{0}; i < rows.size(); ++i) {
      isNonNull[i] = rows[i].has_value();
      if (rows[i].has_value()) {
        nonNullValues.push_back(*rows[i]);
      }
    }

    auto policy = std::make_unique<ManualEncodingSelectionPolicy<int64_t>>(
        std::vector<std::pair<EncodingType, float>>{
            {EncodingType::Trivial, 1.0}},
        CompressionOptions{},
        std::nullopt);
    const auto encoded = EncodingFactory::encodeNullable<int64_t>(
        std::move(policy), nonNullValues, isNonNull, *buffer_);
    return createEncodingView(encoded, pool_.get());
  }
};

TEST_F(NullableEncodingViewTest, readsMixedRows) {
  auto view = makeView({10, std::nullopt, 12, 13, std::nullopt, 15});
  const std::array<uint32_t, 4> indices{1, 2, 4, 5};
  std::array<int64_t, 4> output{};
  std::vector<uint32_t> nullIndices;

  const auto numNonNulls = view->read(
      indices,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      output.data());

  EXPECT_EQ(numNonNulls, 2);
  EXPECT_EQ(output, (std::array<int64_t, 4>{0, 12, 0, 15}));
  EXPECT_EQ(nullIndices, (std::vector<uint32_t>{0, 2}));
}

TEST_F(NullableEncodingViewTest, readsNonNullRows) {
  auto view = makeView({10, std::nullopt, 12, 13, std::nullopt, 15});
  const std::array<uint32_t, 4> indices{5, 0, 2, 2};
  std::array<int64_t, 4> output{};
  std::vector<uint32_t> nullIndices;

  const auto numNonNulls = view->read(
      indices,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      output.data());

  EXPECT_EQ(numNonNulls, indices.size());
  EXPECT_EQ(output, (std::array<int64_t, 4>{15, 10, 12, 12}));
  EXPECT_TRUE(nullIndices.empty());
}

TEST_F(NullableEncodingViewTest, readsNullRows) {
  auto view = makeView({10, std::nullopt, 12, 13, std::nullopt, 15});
  const std::array<uint32_t, 2> indices{1, 4};
  std::array<int64_t, 2> output{};
  std::vector<uint32_t> nullIndices;

  const auto numNonNulls = view->read(
      indices,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      output.data());

  EXPECT_EQ(numNonNulls, 0);
  EXPECT_EQ(output, (std::array<int64_t, 2>{0, 0}));
  EXPECT_EQ(nullIndices, (std::vector<uint32_t>{0, 1}));
}

} // namespace
} // namespace facebook::nimble::test
