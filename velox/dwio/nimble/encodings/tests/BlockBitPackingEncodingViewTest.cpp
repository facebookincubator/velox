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

#include "fmt/core.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using BlockBitPackingEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsBlockBitPackingEncoding) {
  nimble::Vector<int32_t> values{pool_.get()};
  for (auto i = 0; i < 20; ++i) {
    values.push_back(i < 10 ? 500 + i : 9000 + (i % 3));
  }

  nimble::Encoding::Options options;
  options.blockBitPackingBlockSize = 8;
  expectReads<nimble::BlockBitPackingEncoding<int32_t>>(
      values, {17, 0, 8, 9, 19, 3}, options);
}

TEST_F(BlockBitPackingEncodingViewTest, concurrent) {
  nimble::Encoding::Options options;
  options.blockBitPackingBlockSize = 64;

  expectConcurrentReads<nimble::BlockBitPackingEncoding<uint32_t>>(
      randomNarrowUnsigned<uint32_t>(/*seed=*/23),
      randomizedPositions(/*seed=*/24),
      options);
}
