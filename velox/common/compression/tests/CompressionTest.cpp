/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/compression/Compression.h"

namespace facebook::velox::common {

class CompressionTest : public testing::Test {};

TEST_F(CompressionTest, testCompressionNames) {
  EXPECT_EQ("none", compressionKindToString(CompressionKind_NONE));
  EXPECT_EQ("zlib", compressionKindToString(CompressionKind_ZLIB));
  EXPECT_EQ("snappy", compressionKindToString(CompressionKind_SNAPPY));
  EXPECT_EQ("lzo", compressionKindToString(CompressionKind_LZO));
  EXPECT_EQ("lz4", compressionKindToString(CompressionKind_LZ4));
  EXPECT_EQ("zstd", compressionKindToString(CompressionKind_ZSTD));
  EXPECT_EQ(
      "unknown - 99",
      compressionKindToString(static_cast<CompressionKind>(99)));
}

TEST_F(CompressionTest, compressionKindToCodec) {
  ASSERT_EQ(
      folly::io::CodecType::NO_COMPRESSION,
      compressionKindToCodec(CompressionKind::CompressionKind_NONE)->type());
  ASSERT_EQ(
      folly::io::CodecType::LZ4,
      compressionKindToCodec(CompressionKind::CompressionKind_LZ4)->type());
  EXPECT_THROW(
      compressionKindToCodec(CompressionKind::CompressionKind_LZO),
      facebook::velox::VeloxException);
}

TEST_F(CompressionTest, stringToCompressionKind) {
  EXPECT_EQ(stringToCompressionKind("none"), CompressionKind_NONE);
  EXPECT_EQ(stringToCompressionKind("zlib"), CompressionKind_ZLIB);
  EXPECT_EQ(stringToCompressionKind("snappy"), CompressionKind_SNAPPY);
  EXPECT_EQ(stringToCompressionKind("lzo"), CompressionKind_LZO);
  EXPECT_EQ(stringToCompressionKind("lz4"), CompressionKind_LZ4);
  EXPECT_EQ(stringToCompressionKind("zstd"), CompressionKind_ZSTD);
  EXPECT_EQ(stringToCompressionKind("gzip"), CompressionKind_GZIP);
  VELOX_ASSERT_THROW(
      stringToCompressionKind("bz2"), "Not support compression kind bz2");
}
} // namespace facebook::velox::common
