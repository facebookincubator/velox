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

#include "velox/dwio/nimble/common/ChunkHeader.h"

namespace facebook::nimble::test {

TEST(ChunkHeaderTest, size) {
  EXPECT_EQ(kChunkHeaderSize, 5);
  EXPECT_EQ(kChunkHeaderSize, sizeof(uint32_t) + sizeof(CompressionType));
}

TEST(ChunkHeaderTest, writeAndRead) {
  char buffer[kChunkHeaderSize];
  auto* writePos = buffer;
  writeChunkHeader(1'234, CompressionType::Zstd, writePos);
  EXPECT_EQ(writePos - buffer, kChunkHeaderSize);

  const char* readPos = buffer;
  const auto header = readChunkHeader(readPos);
  EXPECT_EQ(readPos - buffer, kChunkHeaderSize);
  EXPECT_EQ(header.length, 1'234);
  EXPECT_EQ(header.compressionType, CompressionType::Zstd);
}

TEST(ChunkHeaderTest, uncompressed) {
  char buffer[kChunkHeaderSize];
  auto* writePos = buffer;
  writeChunkHeader(42, CompressionType::Uncompressed, writePos);

  const char* readPos = buffer;
  const auto header = readChunkHeader(readPos);
  EXPECT_EQ(header.length, 42);
  EXPECT_EQ(header.compressionType, CompressionType::Uncompressed);
}

TEST(ChunkHeaderTest, structuredBinding) {
  char buffer[kChunkHeaderSize];
  auto* writePos = buffer;
  writeChunkHeader(100'000, CompressionType::Zstd, writePos);

  const char* readPos = buffer;
  const auto [length, compressionType] = readChunkHeader(readPos);
  EXPECT_EQ(length, 100'000);
  EXPECT_EQ(compressionType, CompressionType::Zstd);
}

} // namespace facebook::nimble::test
