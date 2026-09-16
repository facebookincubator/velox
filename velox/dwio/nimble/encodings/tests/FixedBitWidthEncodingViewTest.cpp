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

#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using FixedBitWidthEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsFixedBitWidthEncoding) {
  expectReads<nimble::FixedBitWidthEncoding<int32_t>>(
      makeVector({10, 12, 15, 31, 33, 63}), {5, 0, 4, 2, 1});
}

TEST_F(FixedBitWidthEncodingViewTest, concurrent) {
  expectConcurrentReads<nimble::FixedBitWidthEncoding<uint32_t>>(
      randomNarrowUnsigned<uint32_t>(/*seed=*/6),
      randomizedPositions(/*seed=*/7));
}
