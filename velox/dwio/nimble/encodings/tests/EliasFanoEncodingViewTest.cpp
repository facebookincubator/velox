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

#include <cstdint>
#include <limits>
#include <random>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/EliasFanoEncoding.h"

using namespace facebook;

using EliasFanoEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EliasFanoEncodingViewTest, readsSignedAndUnsignedValues) {
  expectReads<nimble::EliasFanoEncoding<int64_t>>(
      makeVector<int64_t>({
          std::numeric_limits<int64_t>::min(),
          -1,
          0,
          0,
          17,
          std::numeric_limits<int64_t>::max(),
      }),
      {5, 0, 3, 1, 3, 4});
  expectReads<nimble::EliasFanoEncoding<uint64_t>>(
      makeVector<uint64_t>({0, 1, 1, 7, 1'000, 1'000, 1'000'000}),
      {6, 0, 2, 3, 5, 1});
}

TEST_F(EliasFanoEncodingViewTest, supportsConcurrentReads) {
  nimble::Vector<uint64_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  uint64_t value{1'000};
  for (uint32_t row{0}; row < kConcurrentRows; ++row) {
    value += row % 7;
    values.push_back(value);
  }

  expectConcurrentReads<nimble::EliasFanoEncoding<uint64_t>>(
      values, randomizedPositions(/*seed=*/43));
}

TEST_F(EliasFanoEncodingViewTest, randomizedReads) {
  std::mt19937_64 random{44};
  std::uniform_int_distribution<uint32_t> rowCountDistribution{1, 512};
  std::uniform_int_distribution<uint64_t> deltaDistribution{0, 1'000};

  for (uint32_t iteration{0}; iteration < 32; ++iteration) {
    SCOPED_TRACE(::testing::Message() << "iteration=" << iteration);
    const auto rowCount = rowCountDistribution(random);
    nimble::Vector<uint64_t> values{pool_.get()};
    values.reserve(rowCount);
    uint64_t value{deltaDistribution(random)};
    for (uint32_t row{0}; row < rowCount; ++row) {
      value += deltaDistribution(random);
      values.push_back(value);
    }

    std::uniform_int_distribution<uint32_t> positionDistribution{
        0, rowCount - 1};
    std::vector<uint32_t> positions;
    positions.reserve(64);
    for (uint32_t read{0}; read < 64; ++read) {
      positions.push_back(positionDistribution(random));
    }
    expectReads<nimble::EliasFanoEncoding<uint64_t>>(values, positions);
  }
}
