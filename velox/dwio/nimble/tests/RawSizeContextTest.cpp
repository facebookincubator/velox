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
#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/writer/RawSizeContext.h"

using namespace facebook;

TEST(RawSizeContextTest, initialColumnCountIsZero) {
  nimble::RawSizeContext ctx;
  EXPECT_EQ(ctx.columnCount(), 0);
}

TEST(RawSizeContextTest, appendSizeIncreasesColumnCount) {
  nimble::RawSizeContext ctx;
  ctx.appendSize(100);
  EXPECT_EQ(ctx.columnCount(), 1);
  ctx.appendSize(200);
  EXPECT_EQ(ctx.columnCount(), 2);
}

TEST(RawSizeContextTest, sizeAtReturnsCorrectValues) {
  nimble::RawSizeContext ctx;
  ctx.appendSize(10);
  ctx.appendSize(20);
  ctx.appendSize(30);
  EXPECT_EQ(ctx.sizeAt(0), 10);
  EXPECT_EQ(ctx.sizeAt(1), 20);
  EXPECT_EQ(ctx.sizeAt(2), 30);
}

TEST(RawSizeContextTest, setSizeAtModifiesValue) {
  nimble::RawSizeContext ctx;
  ctx.appendSize(10);
  ctx.appendSize(20);
  ctx.setSizeAt(0, 99);
  EXPECT_EQ(ctx.sizeAt(0), 99);
  EXPECT_EQ(ctx.sizeAt(1), 20);
}

TEST(RawSizeContextTest, sizeAtOutOfRangeThrows) {
  nimble::RawSizeContext ctx;
  ctx.appendSize(10);
  NIMBLE_ASSERT_THROW(ctx.sizeAt(1), "Column index is out of range.");
}

TEST(RawSizeContextTest, setSizeAtOutOfRangeThrows) {
  nimble::RawSizeContext ctx;
  ctx.appendSize(10);
  NIMBLE_ASSERT_THROW(ctx.setSizeAt(1, 99), "Column index is out of range.");
}

TEST(RawSizeContextTest, getDecodedVectorManagerReturnsUsableReference) {
  nimble::RawSizeContext ctx;
  auto& manager = ctx.getDecodedVectorManager();
  // Verify the manager is functional by acquiring a local decoded vector.
  nimble::DecodedVectorManager::LocalDecodedVector local{manager};
  EXPECT_NE(&local.get(), nullptr);
}
