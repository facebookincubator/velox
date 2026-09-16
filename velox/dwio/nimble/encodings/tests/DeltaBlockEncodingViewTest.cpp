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

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"

using namespace facebook;

using DeltaBlockEncodingViewTest = nimble::test::EncodingViewTest;
using EncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsDeltaBlockEncoding) {
  nimble::Encoding::Options options;
  options.deltaBlockSize = 4;

  expectReads<nimble::DeltaBlockEncoding<int32_t>>(
      makeVector<int32_t>({-10, -9, -9, -1, 0, 2, 2, 17, 18, 40, 41}),
      {10, 0, 3, 4, 7, 8},
      options);
  expectReads<nimble::DeltaBlockEncoding<uint64_t>>(
      makeVector<uint64_t>({0, 0, 1, 5, 9, 9, 100, 150, 151, 1000}),
      {9, 0, 1, 4, 6, 8},
      options);
}

TEST_F(DeltaBlockEncodingViewTest, concurrent) {
  nimble::Encoding::Options options;
  options.deltaBlockSize = 64;

  nimble::Vector<uint64_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  uint64_t value{1000};
  for (uint32_t i = 0; i < kConcurrentRows; ++i) {
    value += 1 + (i % 7);
    values.push_back(value);
  }

  expectConcurrentReads<nimble::DeltaBlockEncoding<uint64_t>>(
      values, randomizedPositions(/*seed=*/41), options);
}
