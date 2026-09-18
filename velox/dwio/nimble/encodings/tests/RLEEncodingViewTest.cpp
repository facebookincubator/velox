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

#include "velox/dwio/nimble/encodings/RLEEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using RLEEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsRleEncoding) {
  expectReads<nimble::RLEEncoding<int32_t>>(
      makeVector({1, 1, 1, 2, 2, 3, 3, 3, 3, 4}), {9, 0, 4, 5, 8, 2});
}

TEST_F(RLEEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/11);

  expectConcurrentReads<nimble::RLEEncoding<int32_t>>(
      randomRleInt32(/*seed=*/12), positions);
  expectConcurrentReads<nimble::RLEEncoding<bool>>(
      randomRleBool(/*seed=*/13), positions);
}
