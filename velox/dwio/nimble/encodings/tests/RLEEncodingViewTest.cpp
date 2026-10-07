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
#include <limits>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/tests/EncodingLayoutTestHelper.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using RLEEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsRleEncoding) {
  expectReads<nimble::RLEEncoding<int32_t>>(
      makeVector({1, 1, 1, 2, 2, 3, 3, 3, 3, 4}), {9, 0, 4, 5, 8, 2});
}

TEST_F(RLEEncodingViewTest, readsPhysicalNestedDeltaBlockValues) {
  const std::vector<int32_t> values{
      0,
      0,
      1,
      1,
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::min(),
      -1,
      -1,
  };
  const nimble::EncodingLayout rleLayout = nimble::RLEEnc{
      .runLengths = nimble::TrivialEnc{},
      .runValues =
          nimble::EncodingLayout{
              nimble::EncodingType::DeltaBlock,
              {},
              nimble::CompressionType::Uncompressed},
  };
  const auto physicalValue = [&](uint32_t row) {
    return std::bit_cast<uint32_t>(values[row]);
  };
  for (const auto useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
    const nimble::Encoding::Options options{
        .useVarintRowCount = useVarint,
        .deltaBlockSize = 4,
    };
    auto policy =
        std::make_unique<nimble::ReplayedEncodingSelectionPolicy<int32_t>>(
            rleLayout,
            nimble::CompressionOptions{},
            [](nimble::DataType type)
                -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
              UNIQUE_PTR_FACTORY(type, nimble::TrivialNestedPolicy);
            });
    const auto encoded = nimble::EncodingFactory::encode<int32_t>(
        std::move(policy), values, *buffer_, options);

    const char* position =
        encoded.data() + nimble::EncodingPrefix::prefixSize(encoded, useVarint);
    const auto runLengthsSize = nimble::encoding::readUint32(position);
    position += runLengthsSize;
    const std::string_view encodedRunValues{
        position, static_cast<size_t>(encoded.end() - position)};
    EXPECT_EQ(
        nimble::EncodingPrefix::encodingType(encodedRunValues),
        nimble::EncodingType::DeltaBlock);
    EXPECT_EQ(
        nimble::EncodingPrefix::readDataType(encodedRunValues),
        nimble::DataType::Uint32);

    const auto view = nimble::createEncodingView(encoded, pool_.get(), options);
    for (uint32_t row = 0; row < values.size(); ++row) {
      SCOPED_TRACE(fmt::format("row={}", row));
      uint32_t actual;
      view->readAt(row, &actual);
      EXPECT_EQ(actual, physicalValue(row));
    }

    const std::vector<uint32_t> positions{8, 0, 4, 5, 5, 6, 2, 9, 1, 7, 3};
    std::vector<uint32_t> actual(positions.size());
    view->readAt(positions, actual.data());
    std::vector<uint32_t> expected;
    expected.reserve(positions.size());
    for (const auto row : positions) {
      expected.push_back(physicalValue(row));
    }
    EXPECT_EQ(actual, expected);

    actual.resize(4);
    view->read(/*offset=*/4, /*length=*/4, actual.data());
    expected.clear();
    for (uint32_t row = 4; row < 8; ++row) {
      expected.push_back(physicalValue(row));
    }
    EXPECT_EQ(actual, expected);
  }
}

TEST_F(RLEEncodingViewTest, preservesFloatingPointRunValues) {
  expectReads<nimble::RLEEncoding<float>>(
      makeVector<float>({1.0F, 1.0F, -0.0F, -0.0F, 2.5F, 2.5F}), {5, 0, 2, 3});
}

TEST_F(RLEEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/11);

  expectConcurrentReads<nimble::RLEEncoding<int32_t>>(
      randomRleInt32(/*seed=*/12), positions);
  expectConcurrentReads<nimble::RLEEncoding<bool>>(
      randomRleBool(/*seed=*/13), positions);
}
