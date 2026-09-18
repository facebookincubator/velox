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
#include <limits>

#include "velox/dwio/nimble/encodings/ALPEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using ALPEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsAlpEncoding) {
  expectReads<nimble::ALPEncoding<float>>(
      makeVector<float>({-12.5F, -1.25F, 0.0F, 1.25F, 12.5F}), {4, 0, 2, 1, 3});
  expectReads<nimble::ALPEncoding<double>>(
      makeVector<double>({-12.5, -1.25, 0.0, 1.25, 12.5}), {4, 0, 2, 1, 3});
  expectReads<nimble::ALPEncoding<float>>(
      makeVector<float>({1.25F, std::numeric_limits<float>::infinity(), 2.5F}),
      {1, 2, 0});
}

TEST_F(ALPEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/14);

  expectConcurrentReads<nimble::ALPEncoding<float>>(
      randomAlpData<float>(/*seed=*/15), positions);
  expectConcurrentReads<nimble::ALPEncoding<double>>(
      randomAlpData<double>(/*seed=*/16), positions);
}
