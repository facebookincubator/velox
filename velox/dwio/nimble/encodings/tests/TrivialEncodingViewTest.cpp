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

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/TrivialEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using TrivialEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsTrivialEncoding) {
  expectReads<nimble::TrivialEncoding<int32_t>>(
      makeVector({10, 20, 30, 40, 50}), {4, 1, 3, 0, 1});
}

TEST_F(EncodingViewTest, readsTrivialEncodingForAllLayoutFamilies) {
  expectReads<nimble::TrivialEncoding<double>>(
      makeVector<double>({1.25, -0.5, 3.0, 1024.125}), {3, 0, 2, 1});
  expectReads<nimble::TrivialEncoding<bool>>(
      makeVector<bool>({true, false, true, true, false}), {4, 0, 1, 3, 2});
  expectReads<nimble::TrivialEncoding<std::string_view>>(
      makeVector<std::string_view>({"alpha", "", "beta", "longer-value"}),
      {3, 0, 1, 2});
}

TEST_F(TrivialEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/2);
  std::vector<std::string> backing;

  expectConcurrentReads<nimble::TrivialEncoding<int32_t>>(
      randomInt32(/*seed=*/3), positions);
  expectConcurrentReads<nimble::TrivialEncoding<bool>>(
      randomBool(/*seed=*/4), positions);
  expectConcurrentReads<nimble::TrivialEncoding<std::string_view>>(
      randomStringViews(backing, /*seed=*/5), positions);
}
