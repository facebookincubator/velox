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

#include "velox/dwio/nimble/encodings/ForEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using FOREncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsForEncoding) {
  expectReads<nimble::ForEncoding<int32_t>>(
      makeVector({100, 101, 102, 103, 5000, 104, 105, 106, -7, -6, -5}),
      {8, 0, 4, 10, 5, 2});
  expectReads<nimble::ForEncoding<uint32_t>>(
      makeVector<uint32_t>({7, 8, 9, 10, 4096, 4097, 4098, 4099}),
      {7, 0, 4, 6, 2});
}

TEST_F(FOREncodingViewTest, concurrent) {
  expectConcurrentReads<nimble::ForEncoding<uint32_t>>(
      randomPforData(/*seed=*/31), randomizedPositions(/*seed=*/32));
}
