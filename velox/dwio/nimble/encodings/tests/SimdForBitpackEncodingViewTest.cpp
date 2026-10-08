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

#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using SimdForBitpackEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsSimdForBitpackEncoding) {
  nimble::Vector<int32_t> values{pool_.get()};
  for (auto i = 0; i < 70; ++i) {
    values.push_back(1000 + (i % 29));
  }
  expectReads<nimble::SimdForBitpackEncoding<int32_t>>(
      values, {69, 0, 31, 32, 64, 7});
}

TEST_F(SimdForBitpackEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/19);

  expectConcurrentReads<nimble::SimdForBitpackEncoding<uint32_t>>(
      randomNarrowUnsigned<uint32_t>(/*seed=*/20), positions);
  expectConcurrentReads<nimble::SimdForBitpackEncoding<uint16_t>>(
      randomNarrowUnsigned<uint16_t>(/*seed=*/21), positions);
  expectConcurrentReads<nimble::SimdForBitpackEncoding<uint8_t>>(
      randomNarrowUnsigned<uint8_t>(/*seed=*/22), positions);
}
