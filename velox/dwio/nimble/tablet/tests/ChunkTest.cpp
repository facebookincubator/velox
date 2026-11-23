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

#include "velox/dwio/nimble/tablet/Chunk.h"

namespace facebook::nimble {
namespace {

TEST(ChunkTest, emptyChunk) {
  Chunk chunk;
  EXPECT_EQ(chunk.rowCount, 0);
  EXPECT_TRUE(chunk.content.empty());
  EXPECT_EQ(chunk.contentSize(), 0);
}

TEST(ChunkTest, singleContent) {
  Chunk chunk;
  chunk.rowCount = 100;
  chunk.content = {"hello"};
  EXPECT_EQ(chunk.contentSize(), 5);
}

TEST(ChunkTest, multipleContent) {
  Chunk chunk;
  chunk.rowCount = 200;
  chunk.content = {"abc", "defgh", "ij"};
  EXPECT_EQ(chunk.contentSize(), 10);
}

TEST(ChunkTest, emptyContent) {
  Chunk chunk;
  chunk.rowCount = 50;
  chunk.content = {"", "", ""};
  EXPECT_EQ(chunk.contentSize(), 0);
}

TEST(ChunkTest, mixedContent) {
  Chunk chunk;
  chunk.rowCount = 10;
  chunk.content = {"data", "", "more"};
  EXPECT_EQ(chunk.contentSize(), 8);
}

} // namespace
} // namespace facebook::nimble
