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

#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using MainlyConstantEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsMainlyConstantEncoding) {
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

TEST_F(MainlyConstantEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/29);

  auto values = constantInt32(5);
  for (uint32_t i = 0; i < values.size(); i += 17) {
    values[i] = static_cast<int32_t>(i);
  }
  expectConcurrentReads<nimble::MainlyConstantEncoding<int32_t>>(
      values, positions);
}
