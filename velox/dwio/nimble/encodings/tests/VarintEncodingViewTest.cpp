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

#include "velox/dwio/nimble/encodings/VarintEncoding.h"

using namespace facebook;

using VarintEncodingViewTest = nimble::test::EncodingViewTest;

TEST_P(VarintEncodingViewTest, readsVarintEncoding) {
  // The smallest value becomes the baseline, and larger values need several
  // varint bytes.
  expectReads<nimble::VarintEncoding<uint32_t>>(
      makeVector<uint32_t>({1'000, 1'000'000, 5, 70'000, 1'000, 300'000'000}),
      {5, 0, 3, 3, 1, 2});
  expectReads<nimble::VarintEncoding<int32_t>>(
      makeVector<int32_t>({1'000, 1'000'000, 5, 70'000, 1'000, 300'000'000}),
      {5, 0, 3, 3, 1, 2});
  expectReads<nimble::VarintEncoding<uint64_t>>(
      makeVector<uint64_t>(
          {1'000, 1'000'000'000'000, 5, 70'000, 1'000, 9'000'000'000'000'000}),
      {5, 0, 3, 3, 1, 2});
  expectReads<nimble::VarintEncoding<int64_t>>(
      makeVector<int64_t>(
          {1'000, 1'000'000'000'000, 5, 70'000, 1'000, 9'000'000'000'000'000}),
      {5, 0, 3, 3, 1, 2});
}

TEST_P(VarintEncodingViewTest, readsAcrossManyValues) {
  expectReads<nimble::VarintEncoding<uint32_t>>(
      randomNarrowUnsigned<uint32_t>(/*seed=*/41),
      randomizedPositions(/*seed=*/42));
}

TEST_P(VarintEncodingViewTest, concurrent) {
  expectConcurrentReads<nimble::VarintEncoding<uint32_t>>(
      randomNarrowUnsigned<uint32_t>(/*seed=*/43),
      randomizedPositions(/*seed=*/44));
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(VarintEncodingViewTest);
