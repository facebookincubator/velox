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

#include "velox/dwio/nimble/encodings/SequenceEncoding.h"

using namespace facebook;

using SequenceEncodingViewTest = nimble::test::EncodingViewTest;

TEST_P(SequenceEncodingViewTest, readsSequenceEncoding) {
  // Steps down through zero.
  expectReads<nimble::SequenceEncoding<int32_t>>(
      makeVector<int32_t>({12, 7, 2, -3, -8, -13, -18, -23}),
      {7, 0, 3, 4, 3, 1});
  // Wraps past the maximum.
  expectReads<nimble::SequenceEncoding<uint64_t>>(
      makeVector<uint64_t>(
          {18'446'744'073'709'551'613ULL,
           18'446'744'073'709'551'614ULL,
           18'446'744'073'709'551'615ULL,
           0,
           1,
           2}),
      {5, 0, 2, 3});
  expectReads<nimble::SequenceEncoding<int8_t>>(
      makeVector<int8_t>({120, 125, -126, -121, -116}), {4, 2, 0});
  // A single row.
  expectReads<nimble::SequenceEncoding<uint16_t>>(
      makeVector<uint16_t>({500}), {0});
}

TEST_P(SequenceEncodingViewTest, concurrent) {
  nimble::Vector<int64_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  for (uint32_t row = 0; row < kConcurrentRows; ++row) {
    values.push_back(-1'000'000 + 3 * int64_t{row});
  }

  expectConcurrentReads<nimble::SequenceEncoding<int64_t>>(
      values, randomizedPositions(/*seed=*/29));
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(SequenceEncodingViewTest);
